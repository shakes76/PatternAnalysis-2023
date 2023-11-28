# dataset:

The ADNI dataset should be placed in this directory.

To ensure that the model and dataset loading works correctly, the dataset should
be added such that it conforms to the following directory structure:

```
    - dataset/:
        - 'test/'
            - 'AD/
                - ...
            - 'NC/'
                - ...
        - 'train/'
            - 'AD/'
                - ...
            - 'NC/'
                - ...
```

'...' represents the ADNI image JPEG files.