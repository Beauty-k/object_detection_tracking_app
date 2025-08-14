import 'package:flutter/material.dart';
import 'package:video_player/video_player.dart';

class VideoResultPlayer extends StatefulWidget {
  final String videoUrl;
  const VideoResultPlayer({super.key, required this.videoUrl});

  @override
  State<VideoResultPlayer> createState() => _VideoResultPlayerState();
}

class _VideoResultPlayerState extends State<VideoResultPlayer> {
  late VideoPlayerController _controller;

  @override
  void initState() {
    super.initState();
    _controller = VideoPlayerController.network(widget.videoUrl)
      ..initialize().then((_) {
        setState(() {});
        _controller.play();
      });
  }

  @override
  Widget build(BuildContext context) {
    return _controller.value.isInitialized
        ? AspectRatio(
            aspectRatio: _controller.value.aspectRatio,
            child: VideoPlayer(_controller),
          )
        : const Center(child: CircularProgressIndicator());
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }
}
