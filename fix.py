#!/usr/bin/env python3

import os
import sys
from pathlib import Path

def patch_memory_store():
    """
    Patches the MemoryStore.add_memory method to handle missing text parameter
    """
    memory_store_path = Path("memory/memory_store.py")
    
    # Make sure the file exists
    if not memory_store_path.exists():
        print(f"Error: {memory_store_path} does not exist")
        return False
    
    # Read the file
    with open(memory_store_path, "r") as f:
        content = f.read()
    
    # Find the add_memory method and check if it's already fixed
    if "def add_memory(self, memory)" in content and "if not hasattr(memory, 'text')" not in content:
        # Add the fix
        fixed_content = content.replace(
            "def add_memory(self, memory):",
            """def add_memory(self, memory, text=None):
        """
        )
        
        fixed_content = fixed_content.replace(
            "# Store text and metadata",
            """# Handle case where memory might not have text attribute
        if not hasattr(memory, 'text') or not memory.text:
            if text is None:
                memory.text = ""
            else:
                memory.text = text
                
        # Store text and metadata"""
        )
        
        # Write the fixed content back
        with open(memory_store_path, "w") as f:
            f.write(fixed_content)
        
        print(f"Successfully patched {memory_store_path}")
        return True
    else:
        print(f"No changes needed for {memory_store_path} (already fixed or structure changed)")
        return False

# Also patch the Memory class to handle missing text
def patch_memory_model():
    """
    Patches the Memory model class to handle missing text
    """
    memory_models_path = Path("memory/models.py")
    
    # Make sure the file exists
    if not memory_models_path.exists():
        print(f"Error: {memory_models_path} does not exist")
        return False
    
    # Read the file
    with open(memory_models_path, "r") as f:
        content = f.read()
    
    # Look for the Memory class definition
    if "class Memory" in content:
        # See if it needs to be patched
        if "text: str = Field(" in content and "default=" not in content:
            # Add default empty string to text field
            fixed_content = content.replace(
                "text: str = Field(",
                "text: str = Field(default='', "
            )
            
            # Write the fixed content back
            with open(memory_models_path, "w") as f:
                f.write(fixed_content)
            
            print(f"Successfully patched {memory_models_path}")
            return True
    
    print(f"No changes needed for {memory_models_path} (already fixed or structure changed)")
    return False

if __name__ == "__main__":
    patch_memory_store()
    patch_memory_model() 