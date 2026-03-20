"""
DISCLAIMER: 
This code was previously part of Joris Heemskerks Bachelors thesis, 
and is being re-used here. All rights are reserved to Joris Heemskerk, 
and Technolution BV, Gouda NL. Joris was granted the rights to use and 
modify this code, at the express notion that a disclaimer was put in.
"""

""" Config yaml template

    This template can be used to validate the composition of the configuration yaml file.
    Minimal viable yaml file looks like (_'s are placeholders for data):

    ```yaml
    general:
        data_images_path : _
        data_annotations_path : _
        input_image_size: _
        grid_size: _
    jobs:
        job0:
            train_val_test_split: _
            batch_size : _
            n_epochs: _
            learning_rate: _
            l1_coefficient: _
            lambda_coord: _
            lambda_noobj: _
            iou_threshold: _
            conf_threshold: _
    ```
"""

CONFIG_TEMPLATE = {
    'type': 'object',
    'properties': {
        'general': {
            'type': 'object',
            'properties': {
                'data_images_path': {
                    'type': 'string', 
                },
                'data_annotations_path': {
                    'type': 'string', 
                },
                'input_image_size': {
                    'type': 'number',
                    'minimum': 0
                },
                'grid_size': {
                    'type': 'number',
                    'minimum': 0
                },
            },
            'required': [
                'data_images_path', 
                'data_annotations_path',
                'input_image_size',
                'grid_size',
            ],
            'additionalProperties' : False
        },
        'jobs': {
            'type': 'object',
            'patternProperties': {
                '^job\\d+$': {
                    'type': 'object',
                    'properties': {
                        'train_val_test_split': {
                            'type': 'array',
                            'items': {'type': 'number'},
                            'minItems': 3,
                            'maxItems': 3
                        },
                        'batch_size': {
                            'type': 'number', 
                            'minimum': 1
                        },
                        'n_epochs': {
                            'type': 'number', 
                            'minimum': 1
                        },
                        'learning_rate': {
                            'type': 'number'
                        },
                        'lambda_coord': {
                            'type': 'number'
                        },
                        'lambda_noobj': {
                            'type': 'number'
                        },
                        'iou_threshold': {
                            'type': 'number'
                        },
                        'conf_threshold': {
                            'type': 'number'
                        }
                    },
                    'required': [
                        'train_val_test_split',
                        'batch_size',
                        'n_epochs',
                        'learning_rate',
                        'lambda_coord',
                        'lambda_noobj',
                        'iou_threshold',
                        'conf_threshold',
                    ],
                    'additionalProperties' : False
                }
            }
        },
    },
    'required': ['general', 'jobs'],
    'additionalProperties' : False
}
