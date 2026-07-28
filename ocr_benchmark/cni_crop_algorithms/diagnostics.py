"""Contrôle local des artefacts intermédiaires générés par les algorithmes."""

from contextvars import ContextVar

GENERATE_INTERMEDIATE_STEPS: ContextVar[bool] = ContextVar(
    "generate_crop_intermediate_steps",
    default=True,
)


def is_required_stage(name: str) -> bool:
    """Conserve la source et la sortie finale même en diagnostic compact."""
    normalized = name.casefold()
    return (
        "source normalis" in normalized
        or "crop" in normalized
        or "perspective corrig" in normalized
        or "meilleur angle" in normalized
    )
