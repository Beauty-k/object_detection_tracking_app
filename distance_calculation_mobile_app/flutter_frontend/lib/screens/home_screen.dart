import 'package:flutter/material.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'Object Distance App',
          style: TextStyle(fontWeight: FontWeight.bold),
        ),
        centerTitle: true,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            ElevatedButton.icon(
              onPressed: () {
                Navigator.pushNamed(context, '/camera');
              },
              icon: const Icon(Icons.camera_alt),
              label: const Text('Capture from Camera'),
                const SizedBox(height: 20),
            ElevatedButton.icon(
              onPressed: () {
                Navigator.pushNamed(context, '/upload');
              },
              icon: const Icon(Icons.video_library),
              label: const Text('Upload Video'),
              style: ElevatedButton.styleFrom(padding: const EdgeInsets.all(16)),
            ),
            const SizedBox(height: 20),
            ElevatedButton.icon(
              onPressed: () {
                Navigator.pushNamed(context, '/youtube');
              },
              icon: const Icon(Icons.link),
              label: const Text('Input YouTube URL'),
              style: ElevatedButton.styleFrom(padding: const EdgeInsets.all(16)),
            ),       style: ElevatedButton.styleFrom(padding: const EdgeInsets.all(16)),
            ),

          ],
        ),
      ),
    );
  }
}

