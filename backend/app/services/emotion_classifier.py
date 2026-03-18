"""Music emotion classification using Music2Emo: valence/arousal → mood descriptions."""

from __future__ import annotations

from pathlib import Path

from app.models import AudioFeatures, LyricEmotion, SegmentEmotion


class EmotionClassifier:
    def __init__(self):
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from music2emo import Music2Emo
            self._model = Music2Emo()
        except Exception as e:
            raise RuntimeError(f"Failed to load Music2Emo model: {e}")

    def classify(
        self, audio_path: Path, audio_features: AudioFeatures
    ) -> list[SegmentEmotion]:
        """Classify emotion for each segment of the audio."""
        self._load_model()

        # Get global emotion prediction
        result = self._model.predict(str(audio_path))
        global_valence = float(result.get("valence", 5.0))
        global_arousal = float(result.get("arousal", 5.0))

        segment_emotions: list[SegmentEmotion] = []
        for i, seg in enumerate(audio_features.segments):
            # Modulate global emotion by segment energy
            energy_ratio = seg.rms_energy / max(
                max(s.rms_energy for s in audio_features.segments), 1e-6
            )
            # Higher energy segments → higher arousal
            seg_arousal = global_arousal + (energy_ratio - 0.5) * 2.0
            seg_arousal = max(1.0, min(9.0, seg_arousal))

            # Spectral centroid affects valence slightly (brighter = more positive)
            max_centroid = max(
                max(s.spectral_centroid for s in audio_features.segments), 1e-6
            )
            brightness = seg.spectral_centroid / max_centroid
            seg_valence = global_valence + (brightness - 0.5) * 1.0
            seg_valence = max(1.0, min(9.0, seg_valence))

            mood = self._mood_description(
                seg_valence, seg_arousal, audio_features.tempo
            )
            visual_mood = self._visual_mood_description(
                seg_valence, seg_arousal, audio_features.tempo
            )

            segment_emotions.append(
                SegmentEmotion(
                    segment_index=i,
                    start=seg.start,
                    end=seg.end,
                    valence=round(seg_valence, 2),
                    arousal=round(seg_arousal, 2),
                    mood_description=mood,
                    visual_mood_description=visual_mood,
                )
            )

        return segment_emotions

    def _mood_description(
        self, valence: float, arousal: float, tempo: float
    ) -> str:
        """Convert valence/arousal to natural language mood description."""
        # Quadrant-based mood
        if valence >= 5.0 and arousal >= 5.0:
            core_mood = "Energetic, joyful, and uplifting"
        elif valence >= 5.0 and arousal < 5.0:
            core_mood = "Peaceful, warm, and serene"
        elif valence < 5.0 and arousal < 5.0:
            core_mood = "Melancholic, introspective, and somber"
        else:
            core_mood = "Intense, dramatic, and dark"

        # Tempo qualifier
        if tempo > 140:
            tempo_q = "fast-paced"
        elif tempo > 100:
            tempo_q = "moderate tempo"
        elif tempo > 70:
            tempo_q = "gentle"
        else:
            tempo_q = "slow and deliberate"

        # Arousal intensity
        if arousal > 7:
            intensity = "building intensity"
        elif arousal > 5:
            intensity = "steady energy"
        elif arousal > 3:
            intensity = "calm flow"
        else:
            intensity = "quiet stillness"

        return f"{core_mood}. {tempo_q.capitalize()} with {intensity}."

    @staticmethod
    def _visual_mood_description(
        valence: float, arousal: float, tempo: float
    ) -> str:
        """Generate a visually descriptive mood prompt optimized for CLIP.

        Uses 8 valence/arousal zones with concrete scene templates.
        Tempo selects which template within each zone for variety.
        """
        # 8 zones: high/low valence × high/mid/low arousal, plus extremes
        if valence >= 6.5 and arousal >= 6.5:
            # High energy, very positive
            scenes = [
                "friends celebrating at a festival with colorful lights and confetti",
                "people dancing on a sunlit beach with turquoise waves",
                "a vibrant city street at night with neon signs and crowds",
                "fireworks exploding over a lively crowd at a summer concert",
            ]
        elif valence >= 5.0 and arousal >= 6.5:
            # High energy, moderately positive
            scenes = [
                "a runner crossing a finish line at sunrise with golden light",
                "waves crashing on rocky cliffs under a dramatic blue sky",
                "a bustling market with bright fruit stalls and sunlight streaming in",
                "cyclists riding through autumn trees with leaves blowing in the wind",
            ]
        elif valence >= 6.5 and arousal < 6.5 and arousal >= 3.5:
            # Moderate energy, very positive
            scenes = [
                "friends walking on a beach at golden hour with warm natural lighting",
                "a cozy cafe with warm lamp light and steaming coffee cups",
                "a sunlit garden with blooming flowers and butterflies",
                "a couple watching a sunset from a hilltop with orange skies",
            ]
        elif valence >= 5.0 and arousal >= 3.5:
            # Moderate energy, moderately positive
            scenes = [
                "a peaceful countryside road lined with green trees in soft daylight",
                "a calm lake reflecting autumn colors under a clear sky",
                "a quiet park bench under cherry blossom trees in spring",
                "a small boat floating on still water at dawn with mist",
            ]
        elif valence < 5.0 and arousal >= 6.5:
            # High energy, negative — intense/dramatic
            scenes = [
                "dark storm clouds gathering over a turbulent ocean",
                "lightning striking a desolate landscape at night",
                "a lone figure standing on a windswept cliff in heavy rain",
                "an abandoned industrial building under a brooding grey sky",
            ]
        elif valence < 3.5 and arousal < 3.5:
            # Low energy, very negative — deep melancholy
            scenes = [
                "an empty bench in a foggy park with fallen leaves",
                "a dimly lit empty room with rain streaking down the window",
                "a solitary tree in a misty field at dusk with fading light",
                "a dark hallway with a single flickering light at the end",
            ]
        elif valence < 5.0 and arousal < 3.5:
            # Low energy, moderately negative — wistful/somber
            scenes = [
                "a quiet rainy street at twilight with reflections on wet pavement",
                "a misty forest path with soft diffused light filtering through trees",
                "an old wooden pier extending into calm grey water under overcast sky",
                "a snow-covered village at dusk with warm light in distant windows",
            ]
        else:
            # Low energy, neutral/slightly negative
            scenes = [
                "a moonlit path through a silent forest with soft blue shadows",
                "a still pond surrounded by willows in the quiet of evening",
                "a deserted street at dawn with pale golden light on old buildings",
                "a distant mountain range under a hazy lavender sky at twilight",
            ]

        # Select scene based on tempo for variety
        idx = int(tempo) % len(scenes)
        scene = scenes[idx]

        # Lighting modifier based on arousal intensity
        if arousal > 7.5:
            lighting = " with dramatic high-contrast lighting"
        elif arousal > 5.5:
            lighting = " with vivid natural lighting"
        elif arousal > 3.5:
            lighting = " with soft warm lighting"
        else:
            lighting = " with dim atmospheric lighting"

        return f"a photograph of {scene}{lighting}"

    @staticmethod
    def enrich_with_lyrics(
        segment_emotions: list[SegmentEmotion],
        lyric_emotions: list[LyricEmotion],
    ) -> list[SegmentEmotion]:
        """Append lyric mood descriptions to audio mood descriptions."""
        lyric_map = {le.segment_index: le for le in lyric_emotions}

        enriched: list[SegmentEmotion] = []
        for se in segment_emotions:
            le = lyric_map.get(se.segment_index)
            if le and le.mood_description:
                new_mood = f"{se.mood_description} The lyrics convey: {le.mood_description}"
                se = se.model_copy(update={"mood_description": new_mood})
            enriched.append(se)

        return enriched
