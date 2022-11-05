# Copyright (c) OpenMMLab. All rights reserved.
"""
Visualize COCO JSON using the OpenMMLab framework
"""
import os
import warnings
from argparse import ArgumentParser
import json
import cv2
from numpy import empty

from mmpose.apis import (get_track_id, inference_top_down_pose_model,
                         init_pose_model, process_mmdet_results,
                         vis_pose_tracking_result)
from mmpose.datasets import DatasetInfo

from mmdet.apis import inference_detector, init_detector

def threewise(iterable):
    a = iter(iterable)
    return zip(a, a, a)

def main():
    parser = ArgumentParser()
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument('--json-path', type=str, help='JSON path')
    parser.add_argument('--video-path', type=str, help='Video path')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show visualizations.')
    parser.add_argument(
        '--out-video-root',
        default='',
        help='Root of the output video file. '
        'Default not saving the visualization video.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=1,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--radius',
        type=int,
        default=4,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')

    args = parser.parse_args()

    assert args.show or (args.out_video_root != '')

    pose_model = init_pose_model(
    args.pose_config, args.pose_checkpoint, device=args.device.lower())

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)


    cap = cv2.VideoCapture(args.video_path)
    fps = None

    assert cap.isOpened(), f'Faild to load video file {args.video_path}'

    if args.out_video_root == '':
        save_out_video = False
    else:
        os.makedirs(args.out_video_root, exist_ok=True)
        save_out_video = True

    if save_out_video:
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter(
            os.path.join(args.out_video_root,
                         f'vis_{os.path.basename(args.video_path)}'), fourcc,
            fps, size)

    frame_id = 0
    track_ids = []
    pose_results = []
    # Opening JSON file
    json_f = open(args.json_path)
    json_data = json.load(json_f)
    results = {}
    results['annotations'] = []

    for data in json_data ["annotations"]:
        track_id = data["track_id"]
        if track_id not in track_ids:
            track_ids.append(track_id)

    while (cap.isOpened()):
        pose_results_last = pose_results

        flag, img = cap.read()
        if not flag:
            break

        pose_results = []
        for data in json_data["annotations"]:
            frame_json_id = data["frame_id"]
            if frame_id != frame_json_id:
                continue
            category_id = data["category_id"] if 'category_id' in data else 1
            if category_id != 1:
                continue
            person = {}
            person['bbox'] = [data["bbox"][0],data["bbox"][1],data["bbox"][0]+data["bbox"][2],data["bbox"][1]+data["bbox"][3],1.0]
            if 'activity' in data:
                person['activity'] = str(data["activity"])
            person['category_id'] = category_id
            person['frame_id'] = frame_id
            person['track_id'] = data["track_id"]
            keypoints =  data["keypoints"]
            person_key_points = []
            for x, y, z in threewise(keypoints):
                person_key_point = [x, y,z]
                person_key_points.append(person_key_point)
            person["keypoints"] = person_key_points
            pose_results.append(person)

        # show the results
        vis_img = vis_pose_tracking_result(
            pose_model,
            img,
            pose_results,
            radius=args.radius,
            thickness=args.thickness,
            dataset=dataset,
            dataset_info=dataset_info,
            kpt_score_thr=args.kpt_thr,
            show=False)

        if args.show:
            cv2.imshow('Image', vis_img)

        if save_out_video:
            videoWriter.write(vis_img)

        if args.show and cv2.waitKey(1) & 0xFF == ord('q'):
            break

        print(f"Frame {frame_id}")
        frame_id += 1

    file_path = "output.json"

    with open(file_path, "w") as fobj:
      json.dump(results, fobj, indent=2)

    cap.release()
    if save_out_video:
        videoWriter.release()
        print("save video")
    if args.show:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
