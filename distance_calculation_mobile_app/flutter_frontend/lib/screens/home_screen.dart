import 'dart:io';
import 'package:flutter/material.dart';

import '../reusable_widgets/custom_app_bar.dart';
import '../reusable_widgets/custom_button.dart';
import '../reusable_widgets/distance_card.dart';
import '../reusable_widgets/video_preview.dart';
import '../reusable_widgets/loading_overlay.dart';
import '../reusable_widgets/video_result_player.dart';
import '../services/file_picker_service.dart';
import '../services/api_service.dart';
import '../helpers/snackbar_message.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  File? selectedVideoFile;
  String apiResponse = "";
  bool isLoading = false;
  String? processedVideoUrl; // Backend processed video URL
  double? measuredDistance; // Distance from backend

  Future<void> pickAndUploadVideo() async {
    final file = await FilePickerService.pickVideoFile();

    if (file == null) {
      showSnackBarMessage(context, "No video selected", isError: true);
      return;
    }

    setState(() {
      selectedVideoFile = file;
      apiResponse = "";
      processedVideoUrl = null;
      measuredDistance = null;
      isLoading = true; // ✅ Show overlay spinner
    });

    try {
      final response = await ApiService.uploadVideoAndCalculateDistance(
        videoFile: file,
        object1: "wallet",
        object2: "card",
      );

      setState(() {
        // Update distance
        measuredDistance = response['distance'] is double
            ? response['distance']
            : null;

        // Update processed video URL
        if (response['video_url'] != null) {
          processedVideoUrl = response['video_url'];
        }

        // Update text message
        apiResponse = measuredDistance != null
            ? "Distance: $measuredDistance mm"
            : "Distance: Not available";
      });

      showSnackBarMessage(context, "Video processed successfully!");
    } catch (e) {
      setState(() {
        apiResponse = "Upload failed: $e";
      });
      showSnackBarMessage(context, "Upload failed: $e", isError: true);
    } finally {
      setState(() {
        isLoading = false; // ✅ Stop overlay spinner
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return LoadingOverlay(
      isLoading: isLoading, // Purple spinner overlay
      child: Scaffold(
        appBar: const CustomAppBar(title: "Distance Measurement"),
        body: SingleChildScrollView(
          child: Column(
            children: [
              const SizedBox(height: 16),

              // Video Preview
              VideoPreview(
                videoPlayer: selectedVideoFile != null
                    ? Text("Selected Video:\n${selectedVideoFile!.path}")
                    : Container(
                        height: 200,
                        color: Colors.grey[300],
                        child: const Center(child: Text("No video selected")),
                      ),
              ),

              const SizedBox(height: 16),

              // Distance Card
              if (measuredDistance != null)
                DistanceCard(
                  object1: "Wallet",
                  object2: "Card",
                  distance: measuredDistance!,
                ),

              const SizedBox(height: 16),

              // Upload Button
              CustomButton(
                label: "Upload Video",
                onPressed: pickAndUploadVideo,
              ),

              const SizedBox(height: 16),

              // API Response
              if (apiResponse.isNotEmpty)
                Text(
                  apiResponse,
                  style: const TextStyle(fontSize: 16, color: Colors.green),
                  textAlign: TextAlign.center,
                ),

              const SizedBox(height: 16),

              // Processed Video Player
              // if (processedVideoUrl != null)
              //   VideoResultPlayer(videoUrl: processedVideoUrl!),
            ],
          ),
        ),
      ),
    );
  }
}
