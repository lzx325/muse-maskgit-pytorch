from functools import partial
def test_function(text, device):
    print(f"Text: {text}, Testing on device: {device}")
class TestClass:
    def __init__(self, device: str):
        self.device = device
        # self.test_function = partial(test_function, device=self.device)
        def test_function_wrapper(text):
            test_function(text, self.device)
        self.test_function = test_function_wrapper

    

if __name__ == '__main__':
    test_class = TestClass(device="cuda")
    test_class.device = "cuda"
    test_class.test_function("test text")