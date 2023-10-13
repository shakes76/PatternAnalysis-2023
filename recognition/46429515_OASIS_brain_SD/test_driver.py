import subprocess
import unittest

class TestGenerativeModel(unittest.TestCase):
    def test_training_script(self):
        # Run the training script (train.py)
        train_process = subprocess.run(["python", "train.py"], capture_output=True, text=True, shell=True)
        
        # Assert that the training script executed successfully (exit code 0)
        self.assertEqual(train_process.returncode, 0)

    def test_prediction_script(self):
        # Run the generating image script (predict.py)
        predict_process = subprocess.run(["python", "predict.py"], capture_output=True, text=True, shell=True)
        
        # Assert that the generating image script executed successfully (exit code 0)
        self.assertEqual(predict_process.returncode, 0)

if __name__ == '__main__':
    unittest.main()
