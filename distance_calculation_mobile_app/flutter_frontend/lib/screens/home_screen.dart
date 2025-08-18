import 'dart:io';
import 'package:flutter/material.dart';
import '../helpers/video_source_picker.dart';
import '../reusable_widgets/custom_app_bar.dart';
import '../reusable_widgets/custom_button.dart';
import '../reusable_widgets/distance_card.dart';
import '../reusable_widgets/video_preview.dart';
import '../reusable_widgets/loading_overlay.dart';
import '../services/api_service.dart';
import '../helpers/snackbar_message.dart';
import '../reusable_widgets/video_result_player.dart';
import '../reusable_widgets/object_name_dialog.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  File? selectedVideoFile;
  String apiResponse = "";
  bool isLoading = false;
  String? processedVideoUrl;
  double? measuredDistance;

  String object1 = "wallet";
  String object2 = "card";

  Future<void> pickAndUploadVideo() async {
    // Ask user to enter object names first
    final objects = await showObjectNameDialog(
      context,
      initialObj1: object1,
      initialObj2: object2,
    );
    if (objects == null) return;

    setState(() {
      object1 = objects[0];
      object2 = objects[1];
    });

    final file = await VideoSourcePicker.show(context);

    if (file == null) {
      showSnackBarMessage(context, "No video selected", isError: true);
      return;
    }

    setState(() {
      selectedVideoFile = file;
      apiResponse = "";
      processedVideoUrl = null;
      measuredDistance = null;
      isLoading = true;
    });

    try {
      final response = await ApiService.uploadVideoAndCalculateDistance(
        videoFile: file,
        object1: object1,
        object2: object2,
      );

      setState(() {
        measuredDistance = response['distance'] is double
            ? response['distance']
            : null;

        if (response['video_url'] != null) {
          processedVideoUrl = response['video_url'];
        }

        // apiResponse = measuredDistance != null
        //     ? "Distance: $measuredDistance mm"
        //     : "Distance: Not available";
      });

      showSnackBarMessage(context, "Video processed successfully!");
    } catch (e) {
      setState(() {
        apiResponse = "Upload failed: $e";
      });
      showSnackBarMessage(context, "Upload failed: $e", isError: true);
    } finally {
      setState(() {
        isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return LoadingOverlay(
      isLoading: isLoading,
      child: Scaffold(
        appBar: const CustomAppBar(title: "Distance Measurement"),
        body: SingleChildScrollView(
          child: Column(
            children: [
              const SizedBox(height: 16),

              VideoPreview(
                videoPlayer: selectedVideoFile != null
                    ? Center(
                        child: SizedBox(
                          height: 250,
                          child: VideoResultPlayer(
                            videoFile: selectedVideoFile!,
                          ),
                        ),
                      )
                    : Container(
                        height: 200,
                        color: Colors.grey[300],
                        child: const Center(child: Text("No video selected")),
                      ),
              ),

              const SizedBox(height: 16),

              if (measuredDistance != null)
                DistanceCard(
                  object1: object1,
                  object2: object2,
                  distance: measuredDistance!,
                ),

              const SizedBox(height: 16),

              CustomButton(
                label: "Upload Video",
                onPressed: pickAndUploadVideo,
              ),

              const SizedBox(height: 16),

              if (apiResponse.isNotEmpty)
                Text(
                  apiResponse,
                  style: const TextStyle(fontSize: 16, color: Colors.green),
                  textAlign: TextAlign.center,
                ),

              const SizedBox(height: 16),

              if (processedVideoUrl != null)
                SizedBox(
                  height: 250,
                  child: VideoResultPlayer(
                    videoUrl: "http://10.0.2.2:8000/static/output.mp4",
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }
}
