import tensorflow as tf
import keras_cv
from keras_cv import bounding_box, visualization


def parse_tfrecord_fn(example):
    """
    A function to take TFRecords, parse them using a description of the features,
    and convert them into a dictionary with the image and bounding boxes as keys.

    Parameters:
    example (TFRecord): Contains information regarding an image and associated bounding boxes

    Returns:
    image_dataset (dictionary): Python dict containing the same information
    """

    feature_description = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    }

    parsed_example = tf.io.parse_single_example(example, feature_description)

    # Decode the JPEG image and normalize the pixel values to the [0, 1] range.
    img = tf.image.decode_jpeg(parsed_example['image/encoded'], channels=3)  # Returned as uint8

    # Get the bounding box coordinates and class labels.
    xmin = tf.sparse.to_dense(parsed_example['image/object/bbox/xmin'])
    xmax = tf.sparse.to_dense(parsed_example['image/object/bbox/xmax'])
    ymin = tf.sparse.to_dense(parsed_example['image/object/bbox/ymin'])
    ymax = tf.sparse.to_dense(parsed_example['image/object/bbox/ymax'])
    labels = tf.sparse.to_dense(parsed_example['image/object/class/label'])

    # Stack the bounding box coordinates to create a [num_boxes, 4] tensor.
    rel_boxes = tf.stack([xmin, ymin, xmax, ymax], axis=-1)
    boxes = bounding_box.convert_format(rel_boxes, source='rel_xyxy', target='xyxy', images=img)

    # Create the final dictionary.
    image_dataset = {
        'images': img,
        'bounding_boxes': {
            'classes': labels,
            'boxes': boxes
        }
    }

    return image_dataset


def visualize_dataset(inputs, value_range, rows, cols, bounding_box_format):
    """
    A function to visualize what a sample of images and bounding boxes looks like. Outputs a
    grid of rows x cols images.

    Parameters:
    inputs (tf.data.Dataset): Contains batched data of images and bounding boxes
    value_range (tuple): Contains the range of pixel values e.g 0-255 or 0-1
    rows (integer): The number of rows you'd like in your grid of images
    cols (integer): The number of cols you'd like in your grid of images
    bounding_box_format (string): Describes the type of bounding box you are dealing with.
    In the case of the plants dataset, this would be 'xyxy'

    Returns:
    None
    """

    inputs = next(iter(inputs.take(1)))
    images, bounding_boxes = inputs["images"], inputs["bounding_boxes"]
    visualization.plot_bounding_box_gallery(
        images,
        value_range=value_range,
        rows=rows,
        cols=cols,
        y_true=bounding_boxes,
        scale=5,
        font_scale=0.7,
        bounding_box_format=bounding_box_format,
        class_mapping=class_mapping,
    )


def visualize_detections(model, dataset, bounding_box_format, rows=2, cols=2):
    """
    A function to take a trained model and visualize predictions of bounding boxes
    given a set of images. Images are presented as grid of rows x cols images.

    Parameters:
    model (tf.keras.Model): Trained object detection model
    dataset (tf.data.Dataset): Contains batched data of images and bounding boxes
    rows (integer): The number of rows you'd like in your grid of images
    cols (integer): The number of cols you'd like in your grid of images
    bounding_box_format (string): Describes the type of bounding box you are dealing with.
    In the case of the plants dataset, this would be 'xyxy'

    Returns:
    None
    """

    images, y_true = next(iter(dataset.take(1)))
    y_pred = model.predict(images)
    y_pred = bounding_box.to_ragged(y_pred)
    visualization.plot_bounding_box_gallery(
        images,
        value_range=(0, 255),
        bounding_box_format=bounding_box_format,
        y_true=y_true,
        y_pred=y_pred,
        scale=4,
        rows=rows,
        cols=cols,
        show=True,
        font_scale=0.7,
        class_mapping=class_mapping,
    )


def dict_to_tuple(inputs):
    """
    A function to take a trained model and visualize predictions of bounding boxes
    given a set of images. Images are presented as grid of rows x cols images.

    Parameters:
    inputs (tf.data.Dataset): Contains batched data of images and bounding boxes

    Returns:
    Tuple of images and associated bounding boxes
    """

    return inputs["images"], bounding_box.to_dense(
        inputs["bounding_boxes"], max_boxes=32
    )


def create_model(config):
    # Building a RetinaNet model with a backbone trained on yolo_v8
    model = keras_cv.models.RetinaNet.from_preset(
        "yolo_v8_m_backbone_coco",
        num_classes=len(class_mapping),
        bounding_box_format=config.bbox_format
    )

    # Customizing non-max supression of model prediction.
    model.prediction_decoder = keras_cv.layers.MultiClassNonMaxSuppression(
        bounding_box_format=config.bbox_format,
        from_logits=True,
        iou_threshold=0.5,
        confidence_threshold=0.5,
    )

    optimizer_Adam = tf.keras.optimizers.legacy.Adam(
        learning_rate=config.base_lr,
        global_clipnorm=10.0
    )

    coco_metrics = keras_cv.metrics.BoxCOCOMetrics(
        bounding_box_format=config.bbox_format, evaluate_freq=5
    )

    # Using focal classification loss and smoothl1 box loss with coco metrics
    model.compile(
        classification_loss=config.classification_loss,
        box_loss=config.box_loss,
        optimizer=optimizer_Adam,
        metrics=[coco_metrics],
        jit_compile=False
    )
    return model


def convert_format_keras_to_wandb(box_list, classes_list):
    """
    Function to convert a bbox and class information from the KerasCV format to
    the WandB format.

    Parameters:
    box_list (list): Information regarding bounding box coordinates in the KerasCV format.
    classes_list (list): Information regarding the class detected.

    Returns:
    Python list with each entry containing a dictionary of a bounding box data in the
    format desired by WandB.
    """
    all_boxes = []
    for b_i, box in enumerate(box_list):
        minX, maxX, minY, maxY = int(box[0]), int(box[2]), int(box[1]), int(box[3])
        class_id = int(classes_list[b_i])
        # get coordinates and labels
        box_data = {
            "position": {
                "minX": minX,
                "maxX": maxX,
                "minY": minY,
                "maxY": maxY},
            "class_id": class_id,
            "box_caption": class_mapping[class_id],
            "domain": "pixel",
        }
        all_boxes.append(box_data)

    return all_boxes


class_mapping = {
    1: 'Apple Scab Leaf',
    2: 'Apple leaf',
    3: 'Apple rust leaf',
    4: 'Bell_pepper leaf',
    5: 'Bell_pepper leaf spot',
    6: 'Blueberry leaf',
    7: 'Cherry leaf',
    8: 'Corn Gray leaf spot',
    9: 'Corn leaf blight',
    10: 'Corn rust leaf',
    11: 'Peach leaf',
    12: 'Potato leaf',
    13: 'Potato leaf early blight',
    14: 'Potato leaf late blight',
    15: 'Raspberry leaf',
    16: 'Soyabean leaf',
    17: 'Soybean leaf',
    18: 'Squash Powdery mildew leaf',
    19: 'Strawberry leaf',
    20: 'Tomato Early blight leaf',
    21: 'Tomato Septoria leaf spot',
    22: 'Tomato leaf',
    23: 'Tomato leaf bacterial spot',
    24: 'Tomato leaf late blight',
    25: 'Tomato leaf mosaic virus',
    26: 'Tomato leaf yellow virus',
    27: 'Tomato mold leaf',
    28: 'Tomato two spotted spider mites leaf',
    29: 'grape leaf',
    30: 'grape leaf black rot'
}
