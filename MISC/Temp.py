class Temp():
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def __len__(self):
        return self.batch_size

    def __getitem__(self, index):
        return index * self.batch_size,(index + 1) * self.batch_size
