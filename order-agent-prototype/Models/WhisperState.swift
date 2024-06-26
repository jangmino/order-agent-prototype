import Foundation
import SwiftUI
import AVFoundation

@MainActor
class WhisperState: NSObject, ObservableObject, AVAudioRecorderDelegate {
    @Published var isModelLoaded = false
    @Published var messageLog = ""
    @Published var transcripedText = ""
    @Published var canTranscribe = false
    @Published var isRecording = false
    
    private var whisperContext: WhisperContext?
    private let recorder = Recorder()
    private var recordedFile: URL? = nil
    private var audioPlayer: AVAudioPlayer?
    private var isdummy: Bool = false
    
    private var modelUrl: URL? {
        Bundle.main.url(forResource: "whisper-small.q4_k", withExtension: "bin", subdirectory: ".")
    }
    
    private enum LoadError: Error {
        case couldNotLocateModel
    }
    
    init(isdummy:Bool = false) {
        super.init()
        self.isdummy = isdummy
    }
    
    func loadModels() async {
        guard let modelUrl = modelUrl, !isdummy else {
            self.messageLog += "WhisperState: Coud not locate model\n"
            return
        }
        await withCheckedContinuation { continuation in
            // 백그라운드 스레드에서 비동기 작업을 수행
            print("start of initialize: \(#file)")
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let whisperContext = try WhisperContext.createContext(path: modelUrl.path())
                    DispatchQueue.main.async {
                        self.whisperContext = whisperContext
                        self.messageLog += "Loaded model \(modelUrl.lastPathComponent)\n"
                        self.canTranscribe = true
                        continuation.resume()
                        print("end of initialize: \(#file)")
                    }
                } catch {
                    DispatchQueue.main.async {
                        self.messageLog += "Could not locate model\n"
                        continuation.resume(throwing: error as! Never)
                    }
                }
            }
        }
    }
    
//    private func loadModel() async throws {
//        messageLog += "Loading model...\n"
//        if let modelUrl {
//            whisperContext = try WhisperContext.createContext(path: modelUrl.path())
//            messageLog += "Loaded model \(modelUrl.lastPathComponent)\n"
//        } else {
//            messageLog += "Could not locate model\n"
//        }
//    }
    
    private func transcribeAudio(_ url: URL) async {
        if (!canTranscribe) {
            return
        }
        guard let whisperContext else {
            return
        }
        
        do {
            canTranscribe = false
            let data = try readAudioSamples(url)
            messageLog += "Transcribing data...\n"
            await whisperContext.fullTranscribe(samples: data)
            let text = await whisperContext.getTranscription()
            messageLog += "Done: \(text)\n"
            transcripedText = text
        } catch {
            print(error.localizedDescription)
            messageLog += "\(error.localizedDescription)\n"
        }
        
        canTranscribe = true
    }
    
    private func readAudioSamples(_ url: URL) throws -> [Float] {
        stopPlayback()
        try startPlayback(url)
        return try decodeWaveFile(url)
    }
    
    func toggleRecord() async {
        transcripedText = ""
        if isRecording {
            await recorder.stopRecording()
            isRecording = false
            if let recordedFile {
                await transcribeAudio(recordedFile)
            }
        } else {
            await requestRecordPermission { granted in
                if granted {
                    Task {
                        do {
                            self.stopPlayback()
                            let file = try FileManager.default.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
                                .appending(path: "output.wav")
                            try await self.recorder.startRecording(toOutputFile: file, delegate: self)
                            self.isRecording = true
                            self.recordedFile = file
                        } catch {
                            print(error.localizedDescription)
                            self.messageLog += "\(error.localizedDescription)\n"
                            self.isRecording = false
                        }
                    }
                }
            }
        }
    }
    
    private func requestRecordPermission(response: @escaping (Bool) -> Void) async {
#if os(macOS)
        response(true)
#else
        if #available(iOS 17.0, *) {
            if await AVAudioApplication.requestRecordPermission() {
                response(true)
            }
        } else {
            // Fallback on earlier versions
            AVAudioSession.sharedInstance().requestRecordPermission { granted in
                response(granted)
            }
        }
#endif
    }
    
    private func startPlayback(_ url: URL) throws {
        audioPlayer = try AVAudioPlayer(contentsOf: url)
        audioPlayer?.play()
    }
    
    private func stopPlayback() {
        audioPlayer?.stop()
        audioPlayer = nil
    }
    
    // MARK: AVAudioRecorderDelegate
    
    nonisolated func audioRecorderEncodeErrorDidOccur(_ recorder: AVAudioRecorder, error: Error?) {
        if let error {
            Task {
                await handleRecError(error)
            }
        }
    }
    
    private func handleRecError(_ error: Error) {
        print(error.localizedDescription)
        messageLog += "\(error.localizedDescription)\n"
        isRecording = false
    }
    
    nonisolated func audioRecorderDidFinishRecording(_ recorder: AVAudioRecorder, successfully flag: Bool) {
        Task {
            await onDidFinishRecording()
        }
    }
    
    private func onDidFinishRecording() {
        isRecording = false
    }
}
