from hamer_wrapper import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequence_folder",
        type=str,
        default=None,
        help="Folder containing the sequence to process",
    )
    parser.add_argument(
        "--use_detector",
        action="store_true",
        help="Use body detector and keypoint detector",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument(
        "--debug", action="store_true", help="Debug mode, show intermediate results"
    )
    args = parser.parse_args()

    if args.sequence_folder is None:
        raise ValueError("Please provide the sequence folder to process")

    sequence_folder = Path(args.sequence_folder)
    loader = MyLoader(sequence_folder)
    rs_serials = loader._rs_serials
    rs_width = loader._rs_width
    rs_height = loader._rs_height
    num_frames = loader._num_frames

    tqdm.write(f">>>>>>>>>> Processing {sequence_folder.name} <<<<<<<<<<<<")

    hamer = HamerWrapper(use_detector=args.use_detector, device=args.device)

    save_folder = sequence_folder / "processed" / "hamer_test"
    save_folder.mkdir(parents=True, exist_ok=True)
    hamer_results = {serial: [None] * num_frames for serial in rs_serials}
    if args.debug:
        vis_images = [None] * num_frames

    for frame_id in tqdm(range(num_frames), ncols=60):
        if args.debug:
            _vis_images = []

        for cam_idx, serial in enumerate(rs_serials):
            img_cv2, boxes, right_flags = loader.get_data_item(serial, frame_id)
            if args.use_detector:
                pred_keypoints_2d_full = hamer.predict(img_cv2)
            else:
                pred_keypoints_2d_full = hamer.predict(img_cv2, boxes, right_flags)
            hamer_results[serial][frame_id] = pred_keypoints_2d_full

            if args.debug:
                vis = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
                for idx, marks in enumerate(pred_keypoints_2d_full):
                    if np.all(marks == -1):
                        continue
                    vis = draw_landmarks(vis, marks, MARK_COLORS[idx])
                _vis_images.append(vis)
        if args.debug:
            vis_images[frame_id] = _vis_images

    tqdm.write("- Saving hamer results...")
    if args.use_detector:
        np.savez_compressed(
            save_folder / "hamer_results_with_detector.npz", **hamer_results
        )
    else:
        np.savez_compressed(save_folder / "hamer_results.npz", **hamer_results)

    if args.debug:
        tqdm.write("- Saving vis images...")
        tqbar = tqdm(total=num_frames, ncols=100)
        video_images = [None] * num_frames
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(
                    draw_image_grid,
                    images=vis_images[frame_id],
                    names=rs_serials,
                    facecolor="black",
                    titlecolor="white",
                ): frame_id
                for frame_id in range(num_frames)
            }
            for future in concurrent.futures.as_completed(futures):
                frame_id = futures[future]
                try:
                    video_images[frame_id] = future.result()
                except Exception as exc:
                    print(f"Error: {exc}")
                tqbar.update(1)
        tqbar.close()

        del vis_images

        # create video
        tqdm.write("- Creating vis video...")
        if args.use_detector:
            video_path = save_folder / "hamer_results_with_detector.mp4"
        else:
            video_path = save_folder / "hamer_results.mp4"
        create_video_from_rgb_images(video_path, video_images, fps=30)

        del video_images

    tqdm.write(f">>>>>>>>>> Done!!! <<<<<<<<<<<<")
