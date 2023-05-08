import 'package:flutter_test/flutter_test.dart';
import 'package:orange_flutter_tflite/orange_flutter_tflite.dart';
import 'package:orange_flutter_tflite/orange_flutter_tflite_platform_interface.dart';
import 'package:orange_flutter_tflite/orange_flutter_tflite_method_channel.dart';
import 'package:plugin_platform_interface/plugin_platform_interface.dart';

class MockOrangeFlutterTflitePlatform
    with MockPlatformInterfaceMixin
    implements OrangeFlutterTflitePlatform {

  @override
  Future<String?> getPlatformVersion() => Future.value('42');
}

void main() {
  final OrangeFlutterTflitePlatform initialPlatform = OrangeFlutterTflitePlatform.instance;

  test('$MethodChannelOrangeFlutterTflite is the default instance', () {
    expect(initialPlatform, isInstanceOf<MethodChannelOrangeFlutterTflite>());
  });

  test('getPlatformVersion', () async {
    // OrangeFlutterTflite orangeFlutterTflitePlugin = OrangeFlutterTflite();
    // MockOrangeFlutterTflitePlatform fakePlatform = MockOrangeFlutterTflitePlatform();
    // OrangeFlutterTflitePlatform.instance = fakePlatform;

    // expect(await orangeFlutterTflitePlugin.getPlatformVersion(), '42');
  });
}
