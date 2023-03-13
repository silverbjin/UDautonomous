#@title Waymo Open Dataset imports
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset import label_pb2
from waymo_open_dataset.metrics.python import keypoint_metrics
from waymo_open_dataset.protos import keypoint_pb2
from waymo_open_dataset.utils import box_utils
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset.utils import keypoint_data
from waymo_open_dataset.utils import keypoint_draw
from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils


# File path to a tfrecods file with Frame protos with human keypoints.
# frame_path = 'frame_with_keypoints.tfrecord'
frame_path = '/home/jin/Documents/UDautonomous/frame_with_keypoints.tfrecord'


#@title Load Frame proto
import tensorflow as tf

dataset = tf.data.TFRecordDataset(frame_path, compression_type='')
for data in dataset:
  frame = dataset_pb2.Frame()
  frame.ParseFromString(bytearray(data.numpy()))
  break

labels = keypoint_data.group_object_labels(frame)
print(f'Loaded {len(labels)} objects')



#@title Auixiliary imports and utils

import os
import math
import numpy as np
from matplotlib import pylab as plt
import plotly.graph_objects as go
import itertools
import PIL.Image
import io
import dataclasses


def _imdecode(buf: bytes) -> np.ndarray:
  with io.BytesIO(buf) as fd:
    pil = PIL.Image.open(fd)
    return np.array(pil)


def _imshow(ax: plt.Axes, image_np: np.ndarray):
  ax.imshow(image_np)
  ax.axis('off')
  ax.set_autoscale_on(False)


def _draw_laser_points(fig: go.Figure,
                       points: np.ndarray,
                       color: str = 'gray',
                       size: int = 3):
  """Visualizes laser points on a plotly figure."""
  fig.add_trace(
      go.Scatter3d(
          mode='markers',
          x=points[:, 0],
          y=points[:, 1],
          z=points[:, 2],
          marker=dict(color=color, size=size)))


def _create_plotly_figure() -> go.Figure:
  """Creates a plotly figure for 3D visualization."""
  fig = go.Figure()
  axis_settings = dict(
      showgrid=False,
      zeroline=False,
      showline=False,
      showbackground=False,
      showaxeslabels=False,
      showticklabels=False)
  fig.update_layout(
      width=600,
      height=600,
      showlegend=False,
      scene=dict(
          aspectmode='data',  # force xyz has same scale,
          xaxis=axis_settings,
          yaxis=axis_settings,
          zaxis=axis_settings,
      ),
  )
  return fig



#@title Select object and camera
object_id = 'DQFLdFau_A8kTPOkDxfgJA'
camera_name = dataset_pb2.CameraName.Name.FRONT_RIGHT

camera_image_by_name = {i.name: i.image for i in frame.images}
obj = labels[object_id]
num_laser_points = len(obj.laser.keypoints.keypoint)
num_camera_points = len(obj.camera[camera_name].keypoints.keypoint)

print(f'Object {object_id} has')
print(f'{num_laser_points} laser keypoints '
      '(short name | location | is_occluded):')
for k in sorted(obj.laser.keypoints.keypoint, key=lambda k: k.type):
  m = k.keypoint_3d.location_m
  location_str = f'({m.x:.2f}, {m.y:.2f}, {m.z:.2f})'
  print(f'{keypoint_draw.point_name(k.type)}\t|'
        f' {location_str:25} | {k.keypoint_3d.visibility.is_occluded}')
print(f'\na LaserKeypoint proto example:\n\n{obj.laser.keypoints.keypoint[0]}')

print(f'{num_camera_points} camera keypoints '
      '(short name |  location | is_occluded):')
for k in sorted(
    obj.camera[camera_name].keypoints.keypoint, key=lambda k: k.type):
  px = k.keypoint_2d.location_px
  location_str = f'({px.x:.0f}, {px.y:.0f})'
  print(f'{keypoint_draw.point_name(k.type)}\t'
        f'| {location_str:13} | {k.keypoint_2d.visibility.is_occluded}')
print(f'\na CameraKeypoint proto example:\n\n'
      f'{obj.camera[camera_name].keypoints.keypoint[0]}')




#@title Show camera keypoints
image_np = _imdecode(camera_image_by_name[camera_name])
croped_image, cropped_camera_keypoints = keypoint_draw.crop_camera_keypoints(
    image_np,
    obj.camera[camera_name].keypoints.keypoint,
    obj.camera[camera_name].box,
    margin=0.3)
camera_wireframe = keypoint_draw.build_camera_wireframe(
    cropped_camera_keypoints)

keypoint_draw.OCCLUDED_BORDER_WIDTH = 3
_, ax = plt.subplots(frameon=False, figsize=(5, 7))
_imshow(ax, croped_image)
keypoint_draw.draw_camera_wireframe(ax, camera_wireframe)




#@title Show laser keypoints

# Select laser points inside pedestrian's bounding box
(range_images, camera_projections, _, range_image_top_pose
) = frame_utils.parse_range_image_and_camera_projection(frame)
points, cp_points = frame_utils.convert_range_image_to_point_cloud(
    frame, range_images, camera_projections, range_image_top_pose)
points_all = np.concatenate(points, axis=0)
box = box_utils.box_to_tensor(obj.laser.box)[tf.newaxis, :]
box_points = points_all[box_utils.is_within_box_3d(points_all, box)[:, 0]]
print(f'{box_points.shape[0]} laser points selected.')

# Visualize 3D scene
laser_wireframe = keypoint_draw.build_laser_wireframe(
    obj.laser.keypoints.keypoint)
fig = _create_plotly_figure()
keypoint_draw.draw_laser_wireframe(fig, laser_wireframe)
_draw_laser_points(fig, box_points)
fig.show()











#@title Example how to compute metrics for camera keypoints
from typing import Tuple

def get_camera_data(
    frame: dataset_pb2.Frame
) -> Tuple[keypoint_data.KeypointsTensors, keypoint_data.KeypointsTensors]:
  """Extracts camera keypoints and bounding boxes from the input Frame proto."""
  all_keypoints = []
  all_boxes = []
  for cl in frame.camera_labels:
    for l in cl.labels:
      if l.HasField('camera_keypoints'):
        box = keypoint_data.create_camera_box_tensors(l.box, dtype=tf.float32)
        keypoints = keypoint_data.create_camera_keypoints_tensors(
            l.camera_keypoints.keypoint,
            default_location=box.center,
            order=keypoint_data.CANONICAL_ORDER_CAMERA,
            dtype=tf.float32)
        all_keypoints.append(keypoints)
        all_boxes.append(box)
  keypoint_tensors = keypoint_data.stack_keypoints(all_keypoints)
  box_tensors = keypoint_data.stack_boxes(all_boxes)
  return keypoint_tensors, box_tensors


gt_cam, gt_cam_box = get_camera_data(frame)

noise_stddev = 5.0  # in pixels
pr_cam = keypoint_data.KeypointsTensors(
    location=gt_cam.location +
    tf.random.normal(gt_cam.location.shape, stddev=noise_stddev),
    visibility=gt_cam.visibility)

all_metrics = keypoint_metrics.create_combined_metric(
    keypoint_metrics.DEFAULT_CONFIG_CAMERA)
all_metrics.update_state([gt_cam, pr_cam, gt_cam_box])
result = all_metrics.result()

print('Camera keypoint metrics:')
for name, tensor in sorted(result.items(), key=lambda e: e[0]):
  print(f'{name:20s}: {tensor.numpy():.3f}')
 
 

#@title Example how to compute metrics for laser keypoints


def get_laser_data(
    frame: dataset_pb2.Frame
) -> Tuple[keypoint_data.KeypointsTensors, keypoint_data.KeypointsTensors]:
  """Extracts laser keypoints and bounding boxes from the input Frame proto."""
  all_keypoints = []
  all_boxes = []
  for l in frame.laser_labels:
    if l.HasField('laser_keypoints'):
      box = keypoint_data.create_laser_box_tensors(l.box, dtype=tf.float32)
      keypoints = keypoint_data.create_laser_keypoints_tensors(
          l.laser_keypoints.keypoint,
          default_location=box.center,
          order=keypoint_data.CANONICAL_ORDER_LASER,
          dtype=tf.float32)
      all_keypoints.append(keypoints)
      all_boxes.append(box)
  keypoint_tensors = keypoint_data.stack_keypoints(all_keypoints)
  box_tensors = keypoint_data.stack_boxes(all_boxes)
  return keypoint_tensors, box_tensors


gt_cam, gt_cam_box = get_laser_data(frame)

noise_stddev = 0.05  # in meters
pr_cam = keypoint_data.KeypointsTensors(
    location=gt_cam.location +
    tf.random.normal(gt_cam.location.shape, stddev=noise_stddev),
    visibility=gt_cam.visibility)

all_metrics = keypoint_metrics.create_combined_metric(
    keypoint_metrics.DEFAULT_CONFIG_LASER)
all_metrics.update_state([gt_cam, pr_cam, gt_cam_box])
result = all_metrics.result()

print('Laser keypoint metrics:')
for name, tensor in sorted(result.items(), key=lambda e: e[0]):
  print(f'{name:20s}: {tensor.numpy():.3f}')
  
  
  
#@title Use individual metrics

per_type_scales = [
    keypoint_metrics.DEFAULT_PER_TYPE_SCALES[t]
    for t in keypoint_data.CANONICAL_ORDER_CAMERA
]
oks = keypoint_metrics.AveragePrecisionAtOKS(per_type_scales, thresholds=[0.95])
oks.update_state([gt_cam, pr_cam, gt_cam_box])
oks.result() 