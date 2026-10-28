# Add project specific ProGuard rules here.
# You can control the set of applied configuration files using the
# proguardFiles setting in build.gradle.
#
# For more details, see
#   http://developer.android.com/guide/developing/tools/proguard.html

# If your project uses WebView with JS, uncomment the following
# and specify the fully qualified class name to the JavaScript interface
# class:
#-keepclassmembers class fqcn.of.javascript.interface.for.webview {
#   public *;
#}

# Uncomment this to preserve the line number information for
# debugging stack traces.
#-keepattributes SourceFile,LineNumberTable

# If you keep the line number information, uncomment this to
# hide the original source file name.
#-renamesourcefileattribute SourceFile

# PyTorch Lite: native code looks up these classes and members by name
# through fbjni, and the AAR ships no consumer rules of its own.
-keep class org.pytorch.** { *; }
-keep class com.facebook.jni.** { *; }
-keep class com.facebook.soloader.nativeloader.** { *; }
-keep @com.facebook.jni.annotations.DoNotStrip class *
-keepclassmembers class * {
    @com.facebook.jni.annotations.DoNotStrip *;
}

# MediaPipe Tasks: the native graph calls back into Java (packets, results,
# options) by name, and its protobuf-lite messages are read reflectively.
-keep class com.google.mediapipe.** { *; }
-keep class com.google.protobuf.** { *; }
# Referenced only by MediaPipe's graph profiler/template APIs, which the AAR
# doesn't ship and this app never calls.
-dontwarn com.google.mediapipe.proto.CalculatorProfileProto$CalculatorProfile
-dontwarn com.google.mediapipe.proto.GraphTemplateProto$CalculatorGraphTemplate

# Flogger (used by MediaPipe) loads its backend with Class.forName, so R8
# can't see it is used; without it MediaPipe's static loggers fail at startup.
-keep class com.google.common.flogger.** { *; }
