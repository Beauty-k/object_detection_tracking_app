import 'dart:io';
import 'package:flutter/material.dart';

import '../reusable_widgets/custom_app_bar.dart';
import '../reusable_widgets/custom_button.dart';
import '../reusable_widgets/distance_card.dart';
import '../reusable_widgets/video_preview.dart';
import '../services/file_picker_service.dart';
import '../services/api_service.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  File? selectedVideoFile;
  String apiResponse = "";
  bool isLoading = false;

  Future<void> pickAndUploadVideo() async {
    final file = await FilePickerService.pickVideoFile();

    if (file == null) {
      ScaffoldMessenger.of(context)
          .showSnackBar(const SnackBar(content: Text("No video selected")));
      return;
    }

    setState(() {
      selectedVideoFile = file;
      apiResponse = "";
      isLoading = true;
    });

    try {
      final response = await ApiService.uploadVideoAndCalculateDistance(
        videoFile: file,
        object1: "wallet",
        object2: "card",
      );

      setState(() {
        apiResponse = "Distance: ${response['distance'] ?? 'N/A'}";
      });
    } catch (e) {
      setState(() {
        apiResponse = "Upload failed: $e";
      });
    } finally {
      setState(() {
        isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: const CustomAppBar(title: "Distance Measurement"),
      body: SingleChildScrollView(
        child: Column(
          children: [
            VideoPreview(
              videoPlayer: selectedVideoFile != null
                  ? Text("Selected Video:\n${selectedVideoFile!.path}")
                  : Container(
                      height: 200,
                      color: Colors.grey[300],
                      child: const Center(child: Text("No video selected")),
                    ),
            ),

            const DistanceCard(
              object1: "Wallet",
              object2: "Card",
              distance: 125.5,
            ),

            const SizedBox(height: 16),

            if (isLoading)
              const CircularProgressIndicator()
            else
              CustomButton(
                label: "Upload Video",
                onPressed: pickAndUploadVideo,
              ),

            const SizedBox(height: 16),

            Text(
              apiResponse,
              style: const TextStyle(fontSize: 16, color: Colors.green),
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
    );
  }
}
