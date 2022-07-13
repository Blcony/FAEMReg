import os
from functools import cmp_to_key


class Data():
    def __init__(self, eval_dir):
        self.eval_dir = eval_dir

    def get_raw_dirs(self):
        root_dir = self.eval_dir
        dirs = []
        seqs = os.listdir(root_dir)
        for seq in seqs:
            seq_dir = os.path.join(root_dir, seq)
            dirs.extend([seq_dir])
        return dirs

    def get_filenames(self):
        data_dirs = self.get_raw_dirs()
        filenames = []
        for dir_path in data_dirs:
            files = os.listdir(dir_path)
            files.sort(key=cmp_to_key(lambda x, y: (int(x.split('.')[0]) - int(y.split('.')[0]))))
            length = len(files)
            for i in range(length):
                fn = os.path.join(dir_path, files[i])
                filenames.append(fn)

        return filenames

    def get_warped_filenames(self):
        data_dirs = self.get_raw_dirs()
        filenames = []
        for dir_path in data_dirs:
            files = os.listdir(dir_path)
            files.sort(key=cmp_to_key(lambda x, y: (int(x.split('_')[0]) - int(y.split('_')[0]))))
            length = len(files)
            for i in range(length):
                fn = os.path.join(dir_path, files[i])
                filenames.append(fn)

        return filenames

