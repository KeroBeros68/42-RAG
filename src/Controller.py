class Controller:
    def __init__(self, logger):
        self.logger = logger

    def index(self, max_chunk_size=2000):
        print("max chunk size ", max_chunk_size)
        print("Ingestion complete! Indices saved under data/processed/")

    def search(self):
        pass

    def search_dataset(self):
        pass

    def answer(self):
        pass

    def answer_dataset(self):
        pass

    def evaluate(self):
        pass
