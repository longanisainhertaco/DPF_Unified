Fine-tune a WALRUS model on DPF training data.

Given $ARGUMENTS (a dataset directory and training parameters), prepare and launch WALRUS fine-tuning on DPF simulation data.

Steps:
1. Validate the dataset using DatasetValidator
2. Describe the WALRUS fine-tuning configuration (learning rate, epochs, batch size)
3. Provide the training command and monitor setup
4. Note: actual training requires `walrus` package and GPU; this command sets up the pipeline

Key files:
- src/dpf/ai/dataset_validator.py — pre-training validation
- src/dpf/ai/well_exporter.py — data format
- src/dpf/ai/field_mapping.py — field conventions

Prerequisites: `pip install dpf-unified[ai]`, WALRUS package, GPU with >= 16GB VRAM.
