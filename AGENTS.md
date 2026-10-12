# AGENTS.md

AutoencoderKL is a research toolkit for training and evaluating autoencoders. Prioritize scientific correctness, extensibility, and reproducible experiments.

## Development principles

- Understand the intended experiment and existing behavior before making changes. Separate implementation improvements from changes to the research method.
- Preserve the meaning of training objectives, latent representations, and evaluation metrics. Make intentional behavioral changes explicit.
- Prefer composable, configuration-driven components. Keep shared abstractions flexible across model families without adding unnecessary complexity.
- Maintain compatibility with existing workflows and saved models unless the requested changes require breaking it. Make any incompatibility explicit.
- Keep experiment settings explicit and comparable. Avoid silently changing defaults that affect reproducibility.

## Workflow and validation

- Follow the project's current environment, tooling, and documentation conventions.
- Validate changes with the smallest meaningful checks. Match the scope of validation to the behavior affected.
- Use controlled comparisons when claiming changes in model quality or performance; record the settings needed to reproduce them.
- Treat resource-intensive training and evaluation as deliberate experiments, rather than routine checks.
- Report what changed, the evidence supporting it, and what remains unverified.
