import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

import 'orange_flutter_tflite_platform_interface.dart';

/// An implementation of [OrangeFlutterTflitePlatform] that uses method channels.
class MethodChannelOrangeFlutterTflite extends OrangeFlutterTflitePlatform {
  /// The method channel used to interact with the native platform.
  @visibleForTesting
  final methodChannel = const MethodChannel('orange_flutter_tflite');

  @override
  Future<String?> getPlatformVersion() async {
    final version = await methodChannel.invokeMethod<String>('getPlatformVersion');
    return version;
  }
}
