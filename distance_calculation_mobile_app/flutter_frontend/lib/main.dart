import 'package:flutter/material.dart';
import 'screens/home_screen.dart';
import 'screens/camera_input_screen.dart';
import 'screens/upload_video_screen.dart';
import 'screens/youtube_input_screen.dart';

void main() {
  runApp(const DistanceApp());
}

class DistanceApp extends StatelessWidget {
  const DistanceApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Object Distance App',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.indigo),
        useMaterial3: true,
      ),
      initialRoute: '/',
      routes: {
        '/': (context) => const HomeScreen(),
        '/camera': (context) => const CameraInputScreen(),
        '/upload': (context) => const UploadVideoScreen(),
        '/youtube': (context) => const YoutubeInputScreen(),
      },
    );
  }
}
