//
//  ContentView.swift
//  order-agent-prototype
//
//  Created by a Developer on 3/6/24.
//

import SwiftUI
import AVFoundation
import whisperllamalib

struct ContentView: View {
    @StateObject var whisperState: WhisperState
    @StateObject var llamaState: LlamaState
        
    private var debug_order_text = "페퍼로니피자 미디엄 사이즈 1개랑 복숭아 티 1잔하구요. 버팔로윙 6개 주세요."
    
    init(isdummy:Bool = false) {
        _whisperState = StateObject(wrappedValue: WhisperState(isdummy: isdummy))
        _llamaState = StateObject(wrappedValue: LlamaState(isdummy: isdummy, model_type: "gemma"))
    }
    
    var body: some View {
        NavigationStack {
            VStack {
                HStack {
                    Button(whisperState.isRecording ? "음성 주문 종료" : "음성 주문 시작", action: {
                        Task {
                            llamaState.generatedText = ""
                            await whisperState.toggleRecord()
                            if whisperState.transcripedText != "" {
//                                await llamaState.complete(text: debug_order_text)
                                await llamaState.complete(text: whisperState.transcripedText)
                            }
                        }
                    })
                    .buttonStyle(.bordered)
                    .font(.system(size: 14, weight: .semibold))
                    .disabled(!whisperState.canTranscribe)
                }
                .padding()
                Divider()
                
                VStack {
                    Text(whisperState.transcripedText)
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundStyle(.blue)
                        .padding()
                    
                    Text(llamaState.generatedText)
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundStyle(.red)
                }
                .frame(minHeight: 200)
                .padding()
                
                Divider()
                
                ScrollView {
                    Text(whisperState.messageLog)
                        .font(.system(size: 12, weight: .semibold))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding()
                }
                .frame(minHeight: 70)
                
                ScrollView(.vertical, showsIndicators: true) {
                    Text(llamaState.messageLog)
                        .font(.system(size: 12, weight: .semibold))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding()
                        .onTapGesture {
                            UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
                        }
                }
                .frame(minHeight: 230)
                
            }
            .navigationTitle("Order Agent Prototype")
            .padding()
        }.onAppear {
            Task {
                await whisperState.loadModels()
                await llamaState.loadModels()
            }
        }
    }

}

#Preview {
    ContentView(isdummy: true)
}
