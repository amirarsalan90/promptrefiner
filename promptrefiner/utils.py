class AbstractLLM:
    def __init__(self):
        pass
    
    def predict(self, input_text):
        """
        Process input_text and return the model's output.
        """
        raise NotImplementedError