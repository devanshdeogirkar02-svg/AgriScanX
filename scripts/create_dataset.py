class PairedDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        w = img.width // 2
        rgb = img.crop((0, 0, w, img.height))
        nir = img.crop((w, 0, img.width, img.height))

        rgb = self.transform(rgb)
        nir = transforms.Grayscale()(self.transform(nir))

        # Augmentations
        if random.random() > 0.5:
            rgb = TF.hflip(rgb)
            nir = TF.hflip(nir)
        if random.random() > 0.5:
            rgb = TF.vflip(rgb)
            nir = TF.vflip(nir)
        if random.random() > 0.6:
            angle = random.choice([90, 180, 270])
            rgb = TF.rotate(rgb, angle)
            nir = TF.rotate(nir, angle)

        return rgb, nir




dataset = PairedDataset(image_paths)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                       num_workers=2, pin_memory=True, drop_last=True)    