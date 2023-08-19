import os

output_dir = './src/bash/'
filename = 'pneumoniamnist_downstream_task.txt'

# Define the possible options for gan_models and gan_steps
gan_models_options = [
    "MHGAN", "SNGAN", "StyleGAN2-D2DCE", "ReACGAN-ADA", "ReACGAN-ADC", "ReACGAN-DiffAug",
    "ACGAN-Mod", "ReACGAN", "BigGAN-DiffAug", "BigGAN-Info", "StyleGAN2-DiffAug",
    "ACGAN-Mod-TAC", "BigGAN", "ReACGAN-TAC", "BigGAN-ADA", "StyleGAN2-Info",
    "ACGAN-Mod-ADC", "StyleGAN2-ADA", "ReACGAN-Info", "StyleGAN2", "ContraGAN", "SAGAN"
]

gan_steps_options = ["20000", "40000", "60000", "80000", "100000"]

# Open a text file for writing
with open(os.path.join(output_dir, filename), "w") as file:

    # 1. synthetic dataset from each GAN (110 exps)
    for model in gan_models_options:
        for step in gan_steps_options:
            file.write(f"dataset_name='pneumoniamnist' gan_models='{model}' gan_steps='{step}'\n")

    # 2. all GANs (1 exp)
    all_models = ",".join(gan_models_options)
    all_steps = ",".join(gan_steps_options)
    file.write(f"dataset_name='pneumoniamnist' gan_models='{all_models}' gan_steps='{all_steps}'\n")

    # 3. all GANs-models for a fixed step (5 exps)
    for step in gan_steps_options:
        file.write(f"dataset_name='pneumoniamnist' gan_models='{all_models}' gan_steps='{step}'\n")

    # 4. Tall GANs-steps for a fixed model (22 exps)
    for model in gan_models_options:
        file.write(f"dataset_name='pneumoniamnist' gan_models='{model}' gan_steps='{all_steps}'\n")

print("File generated successfully!")