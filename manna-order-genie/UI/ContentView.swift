//
//  ContentView.swift
//  manna-order-genie
//
//  Created by 오장민 on 3/6/24.
//

import SwiftUI
import AVFoundation
import whisperlib

struct ContentView: View {
    @StateObject var whisperState = WhisperState()
    
    var body: some View {
        NavigationStack {
            VStack {
                HStack {
                    Button("Transcribe", action: {
                        Task {
                            await whisperState.transcribeSample()
                        }
                    })
                    .buttonStyle(.bordered)
                    .disabled(!whisperState.canTranscribe)
                    
                    Button(whisperState.isRecording ? "Stop recording" : "Start recording", action: {
                        Task {
                            await whisperState.toggleRecord()
                        }
                    })
                    .buttonStyle(.bordered)
                    .disabled(!whisperState.canTranscribe)
                }
                
                ScrollView {
                    Text(verbatim: whisperState.messageLog)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
            .navigationTitle("Whisper SwiftUI Demo")
            .padding()
        }
    }
}

#Preview {
    ContentView()
}
