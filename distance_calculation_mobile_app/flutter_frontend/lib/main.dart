import 'dart:convert';
import 'package:flutter/material.dart';
import 'screens/home_screen.dart';
import 'package:http/http.dart' as http;

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const MyApp());
  testConnection();
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      debugShowCheckedModeBanner: false,
      home: HomeScreen());
  }
}

Future<void> testConnection() async {
  final response = await http.get(Uri.parse("http://10.0.2.2:8000/ping"));

  if (response.statusCode == 200) {
    print("Connected: ${jsonDecode(response.body)}");
  } else {
    print("Connection failed: ${response.statusCode}");
  }
}
