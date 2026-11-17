"""Error type shared by the pipeline stages."""


class PipelineError(Exception):
    """Bad input or a failed stage, with a message meant for the user."""
