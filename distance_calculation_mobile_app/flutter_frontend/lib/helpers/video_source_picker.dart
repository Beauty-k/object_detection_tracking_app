import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import '../services/file_picker_service.dart';
import 'dart:io';

class VideoSourcePicker {
  static Future<File?> show(BuildContext context) async {
    final ImagePicker picker = ImagePicker();

    return showModalBottomSheet<File?>(
      context: context,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(16)),
      ),
      builder: (context) {
        return SafeArea(
          child: Wrap(
            children: [
              ListTile(
                leading: const Icon(Icons.folder_open),
                title: const Text("Pick from Device Storage"),
                onTap: () async {
                  Navigator.pop(context, await FilePickerService.pickVideoFile());
                },
              ),
              ListTile(
                leading: const Icon(Icons.videocam),
                title: const Text("Record Video from Camera"),
                onTap: () async {
                  final XFile? recorded =
                      await picker.pickVideo(source: ImageSource.camera);
                  Navigator.pop(context, recorded != null ? File(recorded.path) : null);
                },
              ),
              ListTile(
                leading: const Icon(Icons.wifi_tethering),
                title: const Text("Live Stream (Coming Soon)"),
                onTap: () {
                  Navigator.pop(context, null);
                  // You can navigate to a live stream screen here
                },
              ),
            ],
          ),
        );
      },
    );
  }
}
