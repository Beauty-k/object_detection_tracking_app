import 'package:flutter/material.dart';

class VideoPreview extends StatelessWidget {
  final Widget videoPlayer; // Pass your video widget here

  const VideoPreview({
    super.key,
    required this.videoPlayer,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(12),
        color: const Color.fromARGB(131, 0, 0, 0),
      ),
      clipBehavior: Clip.antiAlias,
      child: videoPlayer,
    );
  }
}
