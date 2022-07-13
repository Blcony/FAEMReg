import os


class Data():
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def get_raw_dirs(self):
        root_dir = self.data_dir
        dirs = []
        seqs = os.listdir(root_dir)
        for seq in seqs:
            seq_dir = os.path.join(root_dir, seq)
            dirs.extend([seq_dir])
        return dirs
