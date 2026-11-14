"""Natural: pictures as shot. The default, and identical to renders before styles existed."""

from app.styles.base import VideoStyle, register


@register
class Natural(VideoStyle):
    name = "natural"
    label = {"en": "Natural", "zh": "自然"}
    description = {
        "en": "Pictures as shot, with no grading.",
        "zh": "保持原片色彩，不做调色。",
    }
