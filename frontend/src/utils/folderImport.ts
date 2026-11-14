/**
 * Finding the usable images in whatever the user dropped or picked: loose
 * files, a folder, or a folder with folders inside it.
 *
 * Nothing here touches React or the network, so it can be exercised with
 * plain objects.
 */

export const SUPPORTED_EXTENSIONS = ["jpg", "jpeg", "png", "webp"];
export const MAX_IMAGE_BYTES = 50 * 1024 * 1024;
export const MAX_PROJECT_IMAGES = 100;

export interface FoundFile {
  file: File;
  /** Path relative to what was picked, with forward slashes: "trip/day 2/IMG_0042.jpg". */
  path: string;
}

export interface ImportPlan {
  /** The images to upload, in path order. */
  accepted: File[];
  /** How many different folders the accepted images came from. 0 for loose files. */
  folders: number;
  /** Files that are not JPG, PNG or WebP. */
  unsupported: number;
  /** The most common unsupported kinds, upper case, at most four: ["HEIC", "MOV"]. */
  unsupportedKinds: string[];
  /** Supported images over the size limit. */
  tooLarge: number;
  /** Usable images that did not fit in the project. */
  leftOut: number;
}

export function extensionOf(name: string): string {
  const dot = name.lastIndexOf(".");
  return dot > 0 ? name.slice(dot + 1).toLowerCase() : "";
}

/** Files the operating system scatters into folders. Skipped without comment. */
function isJunk(path: string): boolean {
  const parts = path.split("/");
  const name = parts[parts.length - 1].toLowerCase();
  if (name === "thumbs.db" || name === "desktop.ini") return true;
  return parts.some((part) => part.startsWith(".") || part === "__MACOSX");
}

function folderOf(path: string): string {
  const slash = path.lastIndexOf("/");
  return slash >= 0 ? path.slice(0, slash) : "";
}

/**
 * Decide what to upload from the files that were found.
 *
 * `capacity` is how many more images the project can take.
 */
export function planImport(found: FoundFile[], capacity: number): ImportPlan {
  const candidates = found
    .filter((f) => !isJunk(f.path))
    .sort((a, b) => a.path.localeCompare(b.path, undefined, { numeric: true, sensitivity: "base" }));

  const usable: FoundFile[] = [];
  const kinds = new Map<string, number>();
  let unsupported = 0;
  let tooLarge = 0;

  for (const candidate of candidates) {
    const ext = extensionOf(candidate.file.name);
    if (!SUPPORTED_EXTENSIONS.includes(ext)) {
      unsupported += 1;
      const kind = ext ? ext.toUpperCase() : "?";
      kinds.set(kind, (kinds.get(kind) ?? 0) + 1);
    } else if (candidate.file.size > MAX_IMAGE_BYTES) {
      tooLarge += 1;
    } else {
      usable.push(candidate);
    }
  }

  const taken = usable.slice(0, Math.max(0, capacity));
  const folders = new Set(taken.map((f) => folderOf(f.path)).filter((folder) => folder !== ""));

  return {
    accepted: taken.map((f) => f.file),
    folders: folders.size,
    unsupported,
    unsupportedKinds: [...kinds.entries()]
      .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
      .slice(0, 4)
      .map(([kind]) => kind),
    tooLarge,
    leftOut: usable.length - taken.length,
  };
}

/** Files from a file input. A folder input fills in `webkitRelativePath`. */
export function filesFromFileList(list: ArrayLike<File>): FoundFile[] {
  return Array.from(list).map((file) => ({ file, path: file.webkitRelativePath || file.name }));
}

/**
 * The dropped items as file-system entries. Call this inside the drop handler:
 * the browser empties the item list as soon as the handler returns.
 */
export function entriesFromDrop(items: DataTransferItemList): FileSystemEntry[] {
  const entries: FileSystemEntry[] = [];
  for (const item of Array.from(items)) {
    if (item.kind !== "file" || typeof item.webkitGetAsEntry !== "function") continue;
    const entry = item.webkitGetAsEntry();
    if (entry) entries.push(entry);
  }
  return entries;
}

/** A folder's entries. The browser hands them over in batches, so read until one comes back empty. */
function readAll(reader: FileSystemDirectoryReader): Promise<FileSystemEntry[]> {
  return new Promise((resolve, reject) => {
    const all: FileSystemEntry[] = [];
    const next = () =>
      reader.readEntries((batch) => {
        if (batch.length === 0) {
          resolve(all);
        } else {
          all.push(...batch);
          next();
        }
      }, reject);
    next();
  });
}

async function walk(entry: FileSystemEntry, out: FoundFile[]): Promise<void> {
  try {
    if (entry.isFile) {
      const file = await new Promise<File>((resolve, reject) =>
        (entry as FileSystemFileEntry).file(resolve, reject)
      );
      const path = entry.fullPath.startsWith("/") ? entry.fullPath.slice(1) : entry.fullPath;
      out.push({ file, path: path || file.name });
    } else if (entry.isDirectory) {
      const children = await readAll((entry as FileSystemDirectoryEntry).createReader());
      for (const child of children) await walk(child, out);
    }
  } catch {
    // A file or folder that cannot be read is skipped; the rest still imports.
  }
}

/** Every file under the given entries, going into sub-folders. */
export async function filesFromEntries(entries: FileSystemEntry[]): Promise<FoundFile[]> {
  const out: FoundFile[] = [];
  for (const entry of entries) await walk(entry, out);
  return out;
}
