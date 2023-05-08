#
# To learn more about a Podspec see http://guides.cocoapods.org/syntax/podspec.html.
# Run `pod lib lint orange_flutter_tflite.podspec` to validate before publishing.
#
Pod::Spec.new do |s|
  s.name             = 'orange_flutter_tflite'
  s.version          = '0.0.1'
  s.summary          = 'A new Flutter plugin project.'
  s.description      = <<-DESC
A new Flutter plugin project.
                       DESC
  s.homepage         = 'http://example.com'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'Your Company' => 'email@example.com' }
  s.source           = { :path => '.' }
  s.source_files = 'Classes/**/*'
  s.public_header_files = 'Classes/**/*.h'
  s.dependency 'Flutter'
  # s.dependency 'TensorFlowLiteC'
  s.ios.vendored_frameworks = 'TensorFlowLiteC.framework'
  s.platform = :ios, '12.0'

  # Flutter.framework does not contain a i386 slice.
  # s.pod_target_xcconfig = { 'DEFINES_MODULE' => 'YES', 'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386' }
  # s.swift_version = '5.0'
  # s.xcconfig = { 'USER_HEADER_SEARCH_PATHS' => '$(inherited) "${PODS_ROOT}/Headers/Private" "${PODS_ROOT}/Headers/Private/tflite" "${PODS_ROOT}/Headers/Public" "${PODS_ROOT}/Headers/Public/Flutter" "${PODS_ROOT}/Headers/Public/TensorFlowLite/tensorflow_lite" "${PODS_ROOT}/Headers/Public/tflite" "${PODS_ROOT}/TensorFlowLite/Frameworks/tensorflow_lite.framework/Headers" "${PODS_ROOT}/TensorFlowLiteC/Frameworks/TensorFlowLiteC.framework/Headers"' }
  s.pod_target_xcconfig = { 'DEFINES_MODULE' => 'YES', 'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386' }
  s.xcconfig = { 'OTHER_LDFLAGS' => '-framework TensorFlowLiteC -all_load' }
  s.ios.deployment_target = '12.0'
  # s.static_framework = true

  # s.source_files = 'Classes/**/*'
  
  # s.dependency 'Flutter'
  # s.platform = :ios, '12.0'
  # s.ios.vendored_frameworks = 'TensorFlowLiteC.framework'
  

  # Flutter.framework does not contain a i386 slice.
  
end
