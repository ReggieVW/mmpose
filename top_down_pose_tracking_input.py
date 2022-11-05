# Copyright (c) OpenMMLab. All rights reserved.
"""
Adding pose estimations to COCO JSON using the OpenMMLab framework
"""
import os
import warnings
from argparse import ArgumentParser
import json
import cv2
from datetime import datetime, date

from mmpose.apis import (get_track_id, inference_top_down_pose_model,
                         init_pose_model, process_mmdet_results,
                         vis_pose_tracking_result)
from mmpose.datasets import DatasetInfo

from mmdet.apis import inference_detector, init_detector


def main():
    parser = ArgumentParser()
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument('--input-json-path', type=str, help='JSON path')
    parser.add_argument('--output-json-path', type=str, help='JSON path')
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
        '--use-oks-tracking', action='store_true', help='Using OKS tracking')
    parser.add_argument(
        '--tracking-thr', type=float, default=0.3, help='Tracking threshold')
    parser.add_argument(
        '--euro',
        action='store_true',
        help='Using One_Euro_Filter for smoothing')
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

    # optional
    return_heatmap = False

    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    frame_id = 0
    # Opening JSON file
    results = {}
    date_str = f"{date.today():%Y/%m/%d}"
    results['info'] = {"description": os.path.basename(args.video_path), "data_created": date_str}
    results['categories'] = []
    results['annotations'] = []

    key_body_labels = ["nose", "left_eye","right_eye","left_ear","right_ear", "left_shoulder", "right_shoulder",  "left_elbow",
                       "right_elbow",  "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]

    cat_dict_person = {"id": 1, "name": "person",
                       "keypoints": key_body_labels,
                        "skeleton" : [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
                        [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
                        [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],
                        [3, 5], [4, 6]],
                       "supercategory": "person"}
    results["categories"].append(cat_dict_person)

    json_f = open(args.input_json_path)
    json_data = json.load(json_f)

    for data in json_data["annotations"]:
        for data in json_data["categories"]:
            label_name = data["name"]
            if label_name == "person":
                this_person_cat_id = data["id"]

    pose_results = []
    while (cap.isOpened()):
        pose_results_last = pose_results
        flag, img = cap.read()
        if not flag:
            break
        person_results = []
        for data in json_data["annotations"]:
            if data.get("frame_id"):
                frame_json_id = data["frame_id"]
            elif data.get("image_id"):
                frame_json_id = data["image_id"] - 1
            if frame_id != frame_json_id:
                continue
            category_id = data["category_id"]
            if category_id != this_person_cat_id:
                continue
            person = {}
            person['activity'] = ""
            if "activity" in data:
                person['activity'] = data["activity"]
            elif "activity" in data["attributes"]:
                person['activity'] = data["attributes"]["activity"]
            if "track_id" in data:
                person['track_id'] = data["track_id"]
            elif "track_id" in data["attributes"]:
                person['track_id'] = data["attributes"]["track_id"]
            bbox_score = 1.0
            person['bbox'] = [data["bbox"][0],data["bbox"][1],data["bbox"][0]+data["bbox"][2],data["bbox"][1]+data["bbox"][3], bbox_score]
            person['category_id'] = 1
            person_results.append(person)

        # test a single image, with a list of bboxes.
        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            img,
            person_results,
            bbox_thr=args.bbox_thr,
            format='xyxy',
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)


        # get track id for each person instance
        #pose_results, next_id = get_track_id(
        #    pose_results,
        #    pose_results_last,
        #    next_id,
        #    use_oks=args.use_oks_tracking,
        #    tracking_thr=args.tracking_thr,
        #    use_one_euro=args.euro,
        #    fps=fps)

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
            
        for i in range(len(pose_results)):
            pos_arr = pose_results[i]["keypoints"].flatten().tolist()
            key_point = []
            for pos in pos_arr:
                key_point.append(round(pos,3))
            if pose_results[i]["category_id"] ==  1:
                dict_obj = {
                    'track_id': pose_results[i]["track_id"],
                    'frame_id': frame_id,
                    'keypoints': key_point,
                    'bbox': [pose_results[i]["bbox"][0], pose_results[i]["bbox"][1], round(pose_results[i]["bbox"][2]-pose_results[i]["bbox"][0],3),round(pose_results[i]["bbox"][3]-pose_results[i]["bbox"][1],3)],
                    'activity': pose_results[i].get("activity"),
                    'category_id': 1
                    }
            #else:
            #    dict_obj = {
            #        'track_id': pose_results[i]["track_id"],
            #        'frame_id': frame_id,
            #        'keypoints': pose_results[i]["keypoints"].tolist(),
            #        'bbox': pose_results[i]["bbox"].tolist(),
            #        'bbox': [pose_results[i]["bbox"][0], pose_results[i]["bbox"][1], pose_results[i]["bbox"][2]-pose_results[i]["bbox"][0],pose_results[i]["bbox"][3]-pose_results[i]["bbox"][1]],
            #          'category_id': pose_results[i]["category_id"]
            #        }
            results['annotations'].append(dict_obj)

        if args.show:
            cv2.imshow('Image', vis_img)

        if save_out_video:
            videoWriter.write(vis_img)

        if args.show and cv2.waitKey(1) & 0xFF == ord('q'):
            break

        print(f"Frame {frame_id}")
        frame_id += 1

    output_file_path = "annotations.json"
    if args.output_json_path:
        output_file_path = args.output_json_path

    with open(output_file_path, "w") as fobj:
      json.dump(results, fobj, indent=2)

    cap.release()
    if save_out_video:
        videoWriter.release()
        print("save video")
    if args.show:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
