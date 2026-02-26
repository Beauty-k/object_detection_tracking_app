import 'package:flutter/material.dart';

class DistanceCard extends StatelessWidget {
  final String object1;
  final String object2;
  final double distance;
  final String unit;

  const DistanceCard({
    super.key,
    required this.object1,
    required this.object2,
    required this.distance,
    this.unit = "mm",
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      color: Colors.deepPurple.shade50,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      margin: const EdgeInsets.all(12),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              "$object1 ↔ $object2",
              style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
            ),
            const SizedBox(height: 8),
            Text(
              "$distance $unit",
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.bold,
                color: Colors.deepPurple.shade700,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
