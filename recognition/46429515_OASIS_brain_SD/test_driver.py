import subprocess
import unittest
import dataset

class TestGenerativeModel(unittest.TestCase):
    def test_dataset_shape(self):
        # For every image batch:
        # Assert that the batch size is 32
        # Assert that the image channel is equal to 1 (grayscale)
        for step, batch in enumerate(dataset.data_loader):
            self.assertEqual(batch[0][0], 32)
            self.assertEqual(batch[0][1], 1)
    
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
