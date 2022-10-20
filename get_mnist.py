from guided_diffusion_hfai.image_datasets import load_dataset_MNIST_nosampler


temp_data = load_dataset_MNIST_nosampler(train=True, batch_size=64, class_cond=False)
# temp_data = load_dataset_MNIST(train=False, batch_size=64, class_cond=False)
print(temp_data)
print(next(temp_data))