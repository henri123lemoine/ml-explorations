import pickle


class Model:
    def fit(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def save(self):
        pass

    @classmethod
    def load():
        pass

    def save_complete_model(self, file_name: str = None, dir_path: str = "cache", ext: str = "pth"):
        if file_name is None:
            file_name = self.__class__.__name__
        file_path = f"{dir_path}/{file_name}_cls.{ext}"
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_complete_model(cls, file_name: str = None, dir_path: str = "cache", ext: str = "pth"):
        if file_name is None:
            file_name = cls.__name__
        file_path = f"{dir_path}/{file_name}_cls.{ext}"
        with open(file_path, "rb") as f:
            return pickle.load(f)
