import unittest
import os
import tempfile

class SimpleTest(unittest.TestCase):
    """Simple test case that doesn't rely on external dependencies."""
    
    def setUp(self):
        """Set up a test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_file_path = os.path.join(self.temp_dir.name, "test_file.txt")
        
    def tearDown(self):
        """Clean up resources."""
        self.temp_dir.cleanup()
        
    def test_file_operations(self):
        """Test basic file operations."""
        # Write to file
        with open(self.test_file_path, "w") as f:
            f.write("Hello, world!")
            
        # Read from file
        with open(self.test_file_path, "r") as f:
            content = f.read()
            
        # Assert content matches
        self.assertEqual(content, "Hello, world!")
        
    def test_string_operations(self):
        """Test basic string operations."""
        test_string = "  Deep Recall Test  "
        
        # Test strip
        self.assertEqual(test_string.strip(), "Deep Recall Test")
        
        # Test upper/lower
        self.assertEqual(test_string.upper(), "  DEEP RECALL TEST  ")
        self.assertEqual(test_string.lower(), "  deep recall test  ")
        
        # Test split
        self.assertEqual(test_string.split(), ["Deep", "Recall", "Test"])
        
    def test_math_operations(self):
        """Test basic math operations."""
        self.assertEqual(2 + 2, 4)
        self.assertEqual(5 * 5, 25)
        self.assertEqual(10 / 2, 5)
        self.assertAlmostEqual(1 / 3, 0.333333, places=5)
        
    def test_list_operations(self):
        """Test basic list operations."""
        test_list = [1, 2, 3, 4, 5]
        
        # Test append
        test_list.append(6)
        self.assertEqual(test_list, [1, 2, 3, 4, 5, 6])
        
        # Test remove
        test_list.remove(3)
        self.assertEqual(test_list, [1, 2, 4, 5, 6])
        
        # Test slicing
        self.assertEqual(test_list[1:4], [2, 4, 5])
        
    def test_dict_operations(self):
        """Test basic dictionary operations."""
        test_dict = {"name": "Deep Recall", "type": "Memory Framework"}
        
        # Test get
        self.assertEqual(test_dict["name"], "Deep Recall")
        
        # Test update
        test_dict["version"] = "1.0"
        self.assertEqual(test_dict["version"], "1.0")
        
        # Test keys and values
        self.assertEqual(set(test_dict.keys()), {"name", "type", "version"})
        self.assertIn("Memory Framework", test_dict.values())


if __name__ == "__main__":
    unittest.main() 