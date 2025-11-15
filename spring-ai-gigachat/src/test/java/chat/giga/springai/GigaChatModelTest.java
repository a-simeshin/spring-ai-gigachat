package chat.giga.springai;

import static chat.giga.springai.advisor.GigaChatCachingAdvisor.X_SESSION_ID;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.*;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

import chat.giga.springai.advisor.GigaChatCachingAdvisor;
import chat.giga.springai.api.GigaChatInternalProperties;
import chat.giga.springai.api.chat.GigaChatApi;
import chat.giga.springai.api.chat.completion.CompletionRequest;
import chat.giga.springai.api.chat.completion.CompletionResponse;
import chat.giga.springai.api.chat.param.FunctionCallParam;
import chat.giga.springai.tool.GigaTools;
import chat.giga.springai.tool.annotation.GigaTool;
import io.qameta.allure.Description;
import java.util.*;
import java.util.stream.Stream;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.MethodSource;
import org.mockito.ArgumentCaptor;
import org.mockito.ArgumentMatchers;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.ToolResponseMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.metadata.ChatResponseMetadata;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.content.Media;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatusCode;
import org.springframework.http.ResponseEntity;
import org.springframework.util.MimeTypeUtils;
import reactor.core.publisher.Flux;
import reactor.test.StepVerifier;

@ExtendWith(MockitoExtension.class)
public class GigaChatModelTest {
    @Mock
    private GigaChatApi gigaChatApi;

    @Mock
    private GigaChatInternalProperties gigaChatInternalProperties;

    @Mock
    private CompletionResponse response;

    private GigaChatModel gigaChatModel;

    @BeforeEach
    void setUp() {
        gigaChatModel = GigaChatModel.builder()
                .gigaChatApi(gigaChatApi)
                .internalProperties(gigaChatInternalProperties)
                .build();
    }

    @Test
    void testGigaChatOptions_withFunctionCallParam() {
        var functionCallback = GigaTools.from(new TestTool());

        var functionCallParam = FunctionCallParam.builder()
                .name("testToolName")
                .partialArguments(Map.of("arg1", "DEFAULT"))
                .build();

        var prompt = new Prompt(
                List.of(new UserMessage("Hello")),
                GigaChatOptions.builder()
                        .model(GigaChatApi.ChatModel.GIGA_CHAT)
                        .functionCallMode(GigaChatOptions.FunctionCallMode.CUSTOM_FUNCTION)
                        .functionCallParam(functionCallParam)
                        .toolCallbacks(List.of(functionCallback))
                        .build());

        when(gigaChatApi.chatCompletionEntity(any(), any()))
                .thenReturn(new ResponseEntity<>(response, HttpStatusCode.valueOf(200)));

        gigaChatModel.internalCall(prompt, null);

        ArgumentCaptor<CompletionRequest> requestCaptor = ArgumentCaptor.forClass(CompletionRequest.class);
        verify(gigaChatApi).chatCompletionEntity(requestCaptor.capture(), any());

        assertInstanceOf(FunctionCallParam.class, requestCaptor.getValue().getFunctionCall());

        var requestFunctionCallParam =
                (FunctionCallParam) requestCaptor.getValue().getFunctionCall();

        assertEquals(functionCallParam, requestFunctionCallParam);
    }

    @Test
    void testGigaChatOptions_withFunctionCallEmptyAndTool() {
        var functionCallback = GigaTools.from(new TestTool());

        var prompt = new Prompt(
                List.of(new UserMessage("Hello")),
                GigaChatOptions.builder()
                        .model(GigaChatApi.ChatModel.GIGA_CHAT)
                        .toolCallbacks(List.of(functionCallback))
                        .build());

        when(gigaChatApi.chatCompletionEntity(any(), any()))
                .thenReturn(new ResponseEntity<>(response, HttpStatusCode.valueOf(200)));

        gigaChatModel.internalCall(prompt, null);

        ArgumentCaptor<CompletionRequest> requestCaptor = ArgumentCaptor.forClass(CompletionRequest.class);
        verify(gigaChatApi).chatCompletionEntity(requestCaptor.capture(), any());

        assertEquals("auto", requestCaptor.getValue().getFunctionCall());
    }

    @ParameterizedTest
    @EnumSource(GigaChatOptions.FunctionCallMode.class)
    void testGigaChatOptions_withFunctionCallMode(GigaChatOptions.FunctionCallMode callMode) {
        var prompt = new Prompt(
                List.of(new UserMessage("Hello")),
                GigaChatOptions.builder()
                        .model(GigaChatApi.ChatModel.GIGA_CHAT)
                        .functionCallMode(callMode)
                        .build());

        when(gigaChatApi.chatCompletionEntity(any(), any()))
                .thenReturn(new ResponseEntity<>(response, HttpStatusCode.valueOf(200)));

        gigaChatModel.internalCall(prompt, null);

        ArgumentCaptor<CompletionRequest> requestCaptor = ArgumentCaptor.forClass(CompletionRequest.class);
        verify(gigaChatApi).chatCompletionEntity(requestCaptor.capture(), any());

        assertEquals(callMode.getValue(), requestCaptor.getValue().getFunctionCall());
    }

    @ParameterizedTest
    @EnumSource(GigaChatOptions.FunctionCallMode.class)
    void testGigaChatOptions_withFunctionCallModeAndTool(GigaChatOptions.FunctionCallMode callMode) {
        var functionCallback = GigaTools.from(new TestTool());

        var prompt = new Prompt(
                List.of(new UserMessage("Hello")),
                GigaChatOptions.builder()
                        .model(GigaChatApi.ChatModel.GIGA_CHAT)
                        .functionCallMode(callMode)
                        .toolCallbacks(List.of(functionCallback))
                        .build());

        when(gigaChatApi.chatCompletionEntity(any(), any()))
                .thenReturn(new ResponseEntity<>(response, HttpStatusCode.valueOf(200)));

        gigaChatModel.internalCall(prompt, null);

        ArgumentCaptor<CompletionRequest> requestCaptor = ArgumentCaptor.forClass(CompletionRequest.class);
        verify(gigaChatApi).chatCompletionEntity(requestCaptor.capture(), any());

        assertEquals(callMode.getValue(), requestCaptor.getValue().getFunctionCall());
    }

    @Test
    void testGigaChatOptions_withDefault() {
        var prompt = new Prompt(
                List.of(new UserMessage("Hello")),
                GigaChatOptions.builder().model(GigaChatApi.ChatModel.GIGA_CHAT).build());

        when(gigaChatApi.chatCompletionEntity(any(), any()))
                .thenReturn(new ResponseEntity<>(response, HttpStatusCode.valueOf(200)));

        gigaChatModel.internalCall(prompt, null);

        ArgumentCaptor<CompletionRequest> requestCaptor = ArgumentCaptor.forClass(CompletionRequest.class);
        verify(gigaChatApi).chatCompletionEntity(requestCaptor.capture(), any());

        assertNull(requestCaptor.getValue().getFunctionCall());
    }

    @Test
    void testGigaChatOptions_withXSessionID() {
        final var sessionId = "SESSION_ID";
        var prompt = new Prompt(
                List.of(new UserMessage("Hello")),
                GigaChatOptions.builder()
                        .model(GigaChatApi.ChatModel.GIGA_CHAT)
                        .httpHeaders(Map.of(GigaChatCachingAdvisor.X_SESSION_ID, sessionId))
                        .build());

        when(gigaChatApi.chatCompletionEntity(any(), any()))
                .thenReturn(new ResponseEntity<>(response, HttpStatusCode.valueOf(200)));

        gigaChatModel.internalCall(prompt, null);

        ArgumentCaptor<CompletionRequest> requestCaptor = ArgumentCaptor.forClass(CompletionRequest.class);
        ArgumentCaptor<HttpHeaders> headers = ArgumentCaptor.forClass(HttpHeaders.class);
        verify(gigaChatApi).chatCompletionEntity(requestCaptor.capture(), headers.capture());

        assertNull(requestCaptor.getValue().getFunctionCall());
        assertEquals(sessionId, headers.getValue().getFirst(X_SESSION_ID));
    }

    @Test
    void testStream_withToolCall() {
        var spyTestTool = Mockito.spy(new TestTool());

        var prompt = new Prompt(
                List.of(new UserMessage("Hello, test!")),
                GigaChatOptions.builder()
                        .model(GigaChatApi.ChatModel.GIGA_CHAT)
                        .toolCallbacks(GigaTools.from(spyTestTool))
                        .build());

        var functionCallResponse = new CompletionResponse()
                .setId(UUID.randomUUID().toString())
                .setModel(GigaChatApi.ChatModel.GIGA_CHAT.getName())
                .setChoices(List.of(new CompletionResponse.Choice()
                        .setIndex(1)
                        .setFinishReason(CompletionResponse.FinishReason.FUNCTION_CALL)
                        .setDelta(new CompletionResponse.MessagesRes()
                                .setRole(CompletionResponse.Role.assistant)
                                .setContent("")
                                .setFunctionsStateId(UUID.randomUUID().toString())
                                .setFunctionCall(new CompletionResponse.FunctionCall("testMethod", "{}")))));

        // Для первого запроса в гигачат - имитируем вызов функции
        Mockito.when(gigaChatApi.chatCompletionStream(
                        ArgumentMatchers.argThat(
                                rq -> rq != null && rq.getMessages().size() == 1),
                        any()))
                .thenReturn(Flux.just(functionCallResponse));

        var finalResponsePart1 = new CompletionResponse()
                .setId(UUID.randomUUID().toString())
                .setModel(GigaChatApi.ChatModel.GIGA_CHAT.getName())
                .setChoices(List.of(new CompletionResponse.Choice()
                        .setIndex(2)
                        .setDelta(new CompletionResponse.MessagesRes()
                                .setRole(CompletionResponse.Role.assistant)
                                .setContent("Final test response"))));

        var finalResponsePart2 = new CompletionResponse()
                .setId(UUID.randomUUID().toString())
                .setModel(GigaChatApi.ChatModel.GIGA_CHAT.getName())
                .setChoices(List.of(new CompletionResponse.Choice()
                        .setIndex(2)
                        .setDelta(new CompletionResponse.MessagesRes().setContent(""))
                        .setFinishReason(CompletionResponse.FinishReason.STOP)));

        // Для второго запроса в гигачат - имитируем обработку результата вызова функции
        Mockito.when(gigaChatApi.chatCompletionStream(
                        ArgumentMatchers.argThat(rq -> rq.getMessages().size() == 3), any()))
                .thenReturn(Flux.just(finalResponsePart1, finalResponsePart2));

        Flux<ChatResponse> chatResponseFlux = gigaChatModel.stream(prompt);

        // проверяем финальный результат
        StepVerifier.create(chatResponseFlux)
                .assertNext(chatResponse -> {
                    assertNotNull(chatResponse);
                    assertEquals(1, chatResponse.getResults().size());
                    assertEquals(
                            "Final test response",
                            chatResponse.getResults().get(0).getOutput().getText());
                })
                .assertNext(chatResponse -> {
                    assertNotNull(chatResponse);
                    assertEquals(1, chatResponse.getResults().size());
                    assertEquals(
                            "", chatResponse.getResults().get(0).getOutput().getText());
                    assertEquals(
                            "stop",
                            chatResponse.getResults().get(0).getMetadata().getFinishReason());
                })
                .verifyComplete();

        ArgumentCaptor<CompletionRequest> requestCaptor = ArgumentCaptor.forClass(CompletionRequest.class);
        verify(gigaChatApi, times(2)).chatCompletionStream(requestCaptor.capture(), any());

        // Первый запрос в гигачат - с сообщением пользователя и описанием функции testMethod
        var completionRequest1 = requestCaptor.getAllValues().get(0);
        assertEquals(1, completionRequest1.getMessages().size());
        assertEquals(
                CompletionRequest.Role.user,
                completionRequest1.getMessages().get(0).getRole());
        assertEquals("Hello, test!", completionRequest1.getMessages().get(0).getContent());
        assertEquals("auto", completionRequest1.getFunctionCall());
        assertEquals(1, completionRequest1.getFunctions().size());
        assertEquals("testMethod", completionRequest1.getFunctions().get(0).name());

        // Второй запрос в гигачат - с результатом выполнения функции
        var completionRequest2 = requestCaptor.getAllValues().get(1);
        assertEquals(3, completionRequest2.getMessages().size());
        // 1. Сообщение пользователя
        assertEquals(
                CompletionRequest.Role.user,
                completionRequest2.getMessages().get(0).getRole());
        assertEquals("Hello, test!", completionRequest2.getMessages().get(0).getContent());
        // 2. Сообщение ассистента с аргументами для вызова функции
        assertEquals(
                CompletionRequest.Role.assistant,
                completionRequest2.getMessages().get(1).getRole());
        assertEquals("", completionRequest2.getMessages().get(1).getContent());
        assertNotNull(completionRequest2.getMessages().get(1).getFunctionCall());
        // 3. Сообщение с результатом вызова функции
        assertEquals(
                CompletionRequest.Role.function,
                completionRequest2.getMessages().get(2).getRole());
        assertEquals("\"test\"", completionRequest2.getMessages().get(2).getContent());
        assertEquals("auto", completionRequest2.getFunctionCall());
        assertEquals(1, completionRequest2.getFunctions().size());
        assertEquals("testMethod", completionRequest2.getFunctions().get(0).name());

        // Проверяем, что был вызов функции
        verify(spyTestTool).testMethod();
    }

    @Test
    @DisplayName(
            "Тест проверяет, что при вызове чата, если в истории есть два системных промпта, выбрасывается исключение")
    void givenMessagesChatHistoryWithTwoSystemPropmpt_whenSystemPromptSorting_thenThrowIllegalStateException() {
        Prompt prompt = new Prompt(List.of(
                new UserMessage("Какая версия java сейчас актуальна?"),
                new AssistantMessage("23"),
                new SystemMessage("Ты эксперт по работе с  kotlin. Отвечай на вопросы одним словом"),
                new UserMessage("Кто создал Java?"),
                new SystemMessage("Ты эксперт по работе с  java. Отвечай на вопросы одним словом")));
        IllegalStateException exception = assertThrows(IllegalStateException.class, () -> gigaChatModel.call(prompt));

        assertThat(exception.getMessage(), containsStringIgnoringCase("System prompt message must be the only one"));
    }

    private static class TestTool {
        @GigaTool
        public String testMethod() {
            return "test";
        }
    }

    // ======================== Builder and Constructor Tests ========================

    @Test
    @DisplayName("GigaChatModel builder creates model with all parameters")
    @Description("Verify that GigaChatModel can be successfully created using builder pattern with all required parameters. " +
            "Steps: 1. Create mock GigaChatApi, GigaChatInternalProperties, and GigaChatOptions " +
            "2. Build GigaChatModel using builder with all parameters " +
            "3. Verify model is not null and default options are correctly set")
    void testBuilder_withAllParameters() {
        GigaChatApi api = mock(GigaChatApi.class);
        GigaChatInternalProperties props = mock(GigaChatInternalProperties.class);
        GigaChatOptions options = GigaChatOptions.builder().model("test-model").build();

        GigaChatModel model = GigaChatModel.builder()
                .gigaChatApi(api)
                .defaultOptions(options)
                .internalProperties(props)
                .build();

        assertNotNull(model);
        assertEquals(options.getModel(), model.getDefaultOptions().getModel());
    }

    @Test
    @DisplayName("Builder throws exception when GigaChatApi is null")
    @Description("Verify that GigaChatModel builder validates GigaChatApi parameter. " +
            "Steps: 1. Attempt to build GigaChatModel with null GigaChatApi " +
            "2. Verify IllegalArgumentException is thrown")
    void testBuilder_withNullGigaChatApi_shouldThrowException() {
        assertThrows(IllegalArgumentException.class, () -> {
            GigaChatModel.builder()
                    .gigaChatApi(null)
                    .internalProperties(gigaChatInternalProperties)
                    .build();
        });
    }

    @Test
    @DisplayName("Builder throws exception when defaultOptions is null")
    @Description("Verify that GigaChatModel builder validates defaultOptions parameter. " +
            "Steps: 1. Attempt to build GigaChatModel with null defaultOptions " +
            "2. Verify IllegalArgumentException is thrown")
    void testBuilder_withNullDefaultOptions_shouldThrowException() {
        assertThrows(IllegalArgumentException.class, () -> {
            GigaChatModel.builder()
                    .gigaChatApi(gigaChatApi)
                    .defaultOptions(null)
                    .internalProperties(gigaChatInternalProperties)
                    .build();
        });
    }

    @Test
    @DisplayName("Builder throws exception when internalProperties is null")
    @Description("Verify that GigaChatModel builder validates internalProperties parameter. " +
            "Steps: 1. Attempt to build GigaChatModel with null internalProperties " +
            "2. Verify IllegalArgumentException is thrown")
    void testBuilder_withNullInternalProperties_shouldThrowException() {
        assertThrows(IllegalArgumentException.class, () -> {
            GigaChatModel.builder()
                    .gigaChatApi(gigaChatApi)
                    .internalProperties(null)
                    .build();
        });
    }

    // ======================== Models Tests ========================

    @Test
    @DisplayName("Models method returns list of available models")
    @Description("Verify that models() method returns a list of available GigaChat models. " +
            "Steps: 1. Mock GigaChatApi to return ModelsResponse with two models " +
            "2. Call gigaChatModel.models() " +
            "3. Verify response contains 2 models with correct names")
    void testModels_shouldReturnListOfModels() {
        var modelDescription1 = new chat.giga.springai.api.chat.models.ModelDescription(
                "model1", "object", null, null, null);
        var modelDescription2 = new chat.giga.springai.api.chat.models.ModelDescription(
                "model2", "object", null, null, null);

        var modelsResponse = new chat.giga.springai.api.chat.models.ModelsResponse(
                List.of(modelDescription1, modelDescription2));

        when(gigaChatApi.models()).thenReturn(new ResponseEntity<>(modelsResponse, HttpStatusCode.valueOf(200)));

        List<ModelDescription> models = gigaChatModel.models();

        assertNotNull(models);
        assertEquals(2, models.size());
        assertEquals("model1", models.get(0).getModel());
        assertEquals("model2", models.get(1).getModel());
    }

    // ======================== Error Handling Tests ========================

    @Test
    @DisplayName("Call handles null API response gracefully")
    @Description("Verify that call() method handles null response from API correctly. " +
            "Steps: 1. Create prompt with user message " +
            "2. Mock API to return null response " +
            "3. Call gigaChatModel.call() " +
            "4. Verify ChatResponse is not null and has empty results list")
    void testCall_withNullResponse_shouldReturnEmptyList() {
        var prompt = new Prompt(List.of(new UserMessage("Hello")));

        when(gigaChatApi.chatCompletionEntity(any(), any()))
                .thenReturn(new ResponseEntity<>(null, HttpStatusCode.valueOf(200)));

        ChatResponse chatResponse = gigaChatModel.call(prompt);

        assertNotNull(chatResponse);
        assertEquals(0, chatResponse.getResults().size());
    }

    @Test
    @DisplayName("Call handles empty choices list gracefully")
    @Description("Verify that call() method handles empty choices list in API response. " +
            "Steps: 1. Create prompt with user message " +
            "2. Mock API to return CompletionResponse with empty choices " +
            "3. Call gigaChatModel.call() " +
            "4. Verify ChatResponse has empty results list")
    void testCall_withEmptyChoices_shouldHandleGracefully() {
        var prompt = new Prompt(List.of(new UserMessage("Hello")));
        var emptyResponse = new CompletionResponse()
                .setId(UUID.randomUUID().toString())
                .setModel(GigaChatApi.ChatModel.GIGA_CHAT.getName())
                .setChoices(List.of());

        when(gigaChatApi.chatCompletionEntity(any(), any()))
                .thenReturn(new ResponseEntity<>(emptyResponse, HttpStatusCode.valueOf(200)));

        ChatResponse chatResponse = gigaChatModel.call(prompt);

        assertNotNull(chatResponse);
        assertEquals(0, chatResponse.getResults().size());
    }

    @Test
    @DisplayName("Stream propagates API errors")
    @Description("Verify that stream() method properly propagates errors from API. " +
            "Steps: 1. Create prompt with user message " +
            "2. Mock API to return Flux with error " +
            "3. Call gigaChatModel.stream() " +
            "4. Verify error is propagated with correct message")
    void testStream_withError_shouldPropagateError() {
        var prompt = new Prompt(List.of(new UserMessage("Hello")));
        var error = new RuntimeException("API Error");

        when(gigaChatApi.chatCompletionStream(any(), any())).thenReturn(Flux.error(error));

        Flux<ChatResponse> chatResponseFlux = gigaChatModel.stream(prompt);

        StepVerifier.create(chatResponseFlux)
                .expectErrorMatches(throwable -> throwable instanceof RuntimeException
                        && throwable.getMessage().equals("API Error"))
                .verify();
    }

    // ======================== Media Upload Tests ========================

    @Test
    @DisplayName("New media file is uploaded and ID is set")
    @Description("Verify that new media without ID is uploaded to GigaChat and ID is assigned. " +
            "Steps: 1. Create media without ID " +
            "2. Create prompt with UserMessage containing media " +
            "3. Mock uploadFile to return upload response with ID " +
            "4. Call gigaChatModel.call() " +
            "5. Verify uploadFile was called and media ID is in request attachments")
    void testUploadMedia_withNewMedia_shouldUploadAndSetId() {
        var mediaId = UUID.randomUUID();
        var media = Media.builder()
                .data("test data")
                .mimeType(MimeTypeUtils.IMAGE_PNG)
                .build();

        var prompt = new Prompt(List.of(
                UserMessage.builder().text("Test").media(media).build()));

        var uploadResponse = new chat.giga.springai.api.chat.file.UploadFileResponse(
                null, null, null, mediaId, null, null, null);

        when(gigaChatApi.uploadFile(any())).thenReturn(new ResponseEntity<>(uploadResponse, HttpStatusCode.valueOf(200)));
        when(gigaChatApi.chatCompletionEntity(any(), any()))
                .thenReturn(new ResponseEntity<>(response, HttpStatusCode.valueOf(200)));

        gigaChatModel.call(prompt);

        ArgumentCaptor<CompletionRequest> requestCaptor = ArgumentCaptor.forClass(CompletionRequest.class);
        verify(gigaChatApi).chatCompletionEntity(requestCaptor.capture(), any());
        verify(gigaChatApi).uploadFile(media);

        assertEquals(1, requestCaptor.getValue().getMessages().get(0).getAttachments().size());
        assertEquals(mediaId, requestCaptor.getValue().getMessages().get(0).getAttachments().get(0));
    }

    @Test
    @DisplayName("Media with existing ID is not uploaded again")
    @Description("Verify that media with existing ID is not uploaded to GigaChat again. " +
            "Steps: 1. Create media with existing ID " +
            "2. Create prompt with UserMessage containing media " +
            "3. Call gigaChatModel.call() " +
            "4. Verify uploadFile was never called")
    void testUploadMedia_withExistingId_shouldNotUploadAgain() {
        var existingMediaId = UUID.randomUUID().toString();
        var media = Media.builder()
                .id(existingMediaId)
                .data("test data")
                .mimeType(MimeTypeUtils.IMAGE_PNG)
                .build();

        var prompt = new Prompt(List.of(
                UserMessage.builder().text("Test").media(media).build()));

        when(gigaChatApi.chatCompletionEntity(any(), any()))
                .thenReturn(new ResponseEntity<>(response, HttpStatusCode.valueOf(200)));

        gigaChatModel.call(prompt);

        verify(gigaChatApi, never()).uploadFile(any());
    }

    @Test
    @DisplayName("Multiple media files are uploaded")
    @Description("Verify that multiple media files without IDs are uploaded to GigaChat. " +
            "Steps: 1. Create two media objects without IDs " +
            "2. Create prompt with UserMessage containing both media " +
            "3. Mock uploadFile for each media " +
            "4. Call gigaChatModel.call() " +
            "5. Verify uploadFile was called for both media")
    void testUploadMedia_withMultipleMedia_shouldUploadAll() {
        var mediaId1 = UUID.randomUUID();
        var mediaId2 = UUID.randomUUID();

        var media1 = Media.builder()
                .data("test data 1")
                .mimeType(MimeTypeUtils.IMAGE_PNG)
                .build();
        var media2 = Media.builder()
                .data("test data 2")
                .mimeType(MimeTypeUtils.IMAGE_JPEG)
                .build();

        var prompt = new Prompt(List.of(
                UserMessage.builder().text("Test").media(List.of(media1, media2)).build()));

        when(gigaChatApi.uploadFile(media1))
                .thenReturn(new ResponseEntity<>(
                        new chat.giga.springai.api.chat.file.UploadFileResponse(
                                null, null, null, mediaId1, null, null, null),
                        HttpStatusCode.valueOf(200)));
        when(gigaChatApi.uploadFile(media2))
                .thenReturn(new ResponseEntity<>(
                        new chat.giga.springai.api.chat.file.UploadFileResponse(
                                null, null, null, mediaId2, null, null, null),
                        HttpStatusCode.valueOf(200)));
        when(gigaChatApi.chatCompletionEntity(any(), any()))
                .thenReturn(new ResponseEntity<>(response, HttpStatusCode.valueOf(200)));

        gigaChatModel.call(prompt);

        verify(gigaChatApi).uploadFile(media1);
        verify(gigaChatApi).uploadFile(media2);
    }

    // ======================== Usage/Tokens Tests ========================

    @Test
    @DisplayName("Usage is calculated correctly for single call")
    @Description("Verify that token usage is correctly calculated for a single chat completion call. " +
            "Steps: 1. Create prompt with user message " +
            "2. Mock API to return response with usage data (10 prompt, 20 completion, 30 total tokens) " +
            "3. Call gigaChatModel.call() " +
            "4. Verify usage metadata has correct token counts")
    void testUsageCalculation_withSingleCall() {
        var prompt = new Prompt(List.of(new UserMessage("Hello")));
        var usage = new CompletionResponse.Usage(10, 20, 30);
        var completionResponse = new CompletionResponse()
                .setId(UUID.randomUUID().toString())
                .setModel(GigaChatApi.ChatModel.GIGA_CHAT.getName())
                .setUsage(usage)
                .setChoices(List.of(new CompletionResponse.Choice()
                        .setIndex(0)
                        .setMessage(new CompletionResponse.MessagesRes()
                                .setRole(CompletionResponse.Role.assistant)
                                .setContent("Hi"))));

        when(gigaChatApi.chatCompletionEntity(any(), any()))
                .thenReturn(new ResponseEntity<>(completionResponse, HttpStatusCode.valueOf(200)));

        ChatResponse chatResponse = gigaChatModel.call(prompt);

        assertNotNull(chatResponse.getMetadata().getUsage());
        assertEquals(10L, chatResponse.getMetadata().getUsage().getPromptTokens());
        assertEquals(20L, chatResponse.getMetadata().getUsage().getGenerationTokens());
        assertEquals(30L, chatResponse.getMetadata().getUsage().getTotalTokens());
    }

    @Test
    @DisplayName("Null usage is handled gracefully")
    @Description("Verify that null usage in API response is handled without errors. " +
            "Steps: 1. Create prompt with user message " +
            "2. Mock API to return response with null usage " +
            "3. Call gigaChatModel.call() " +
            "4. Verify usage metadata is not null")
    void testUsageCalculation_withNullUsage() {
        var prompt = new Prompt(List.of(new UserMessage("Hello")));
        var completionResponse = new CompletionResponse()
                .setId(UUID.randomUUID().toString())
                .setModel(GigaChatApi.ChatModel.GIGA_CHAT.getName())
                .setUsage(null)
                .setChoices(List.of(new CompletionResponse.Choice()
                        .setIndex(0)
                        .setMessage(new CompletionResponse.MessagesRes()
                                .setRole(CompletionResponse.Role.assistant)
                                .setContent("Hi"))));

        when(gigaChatApi.chatCompletionEntity(any(), any()))
                .thenReturn(new ResponseEntity<>(completionResponse, HttpStatusCode.valueOf(200)));

        ChatResponse chatResponse = gigaChatModel.call(prompt);

        assertNotNull(chatResponse.getMetadata().getUsage());
    }

    // ======================== Message Conversion Tests ========================

    @Test
    @DisplayName("UserMessage is converted correctly")
    @Description("Verify that UserMessage is correctly converted to CompletionRequest.Message with user role. " +
            "Steps: 1. Create prompt with UserMessage " +
            "2. Call gigaChatModel.call() " +
            "3. Verify CompletionRequest has message with user role and correct content")
    void testMessageConversion_userMessage() {
        var prompt = new Prompt(List.of(new UserMessage("Test message")));

        when(gigaChatApi.chatCompletionEntity(any(), any()))
                .thenReturn(new ResponseEntity<>(response, HttpStatusCode.valueOf(200)));

        gigaChatModel.call(prompt);

        ArgumentCaptor<CompletionRequest> requestCaptor = ArgumentCaptor.forClass(CompletionRequest.class);
        verify(gigaChatApi).chatCompletionEntity(requestCaptor.capture(), any());

        assertEquals(1, requestCaptor.getValue().getMessages().size());
        assertEquals(CompletionRequest.Role.user, requestCaptor.getValue().getMessages().get(0).getRole());
        assertEquals("Test message", requestCaptor.getValue().getMessages().get(0).getContent());
    }

    @Test
    @DisplayName("SystemMessage is converted correctly")
    @Description("Verify that SystemMessage is correctly converted to CompletionRequest.Message with system role. " +
            "Steps: 1. Create prompt with SystemMessage " +
            "2. Call gigaChatModel.call() " +
            "3. Verify CompletionRequest has message with system role and correct content")
    void testMessageConversion_systemMessage() {
        var prompt = new Prompt(List.of(new SystemMessage("System prompt")));

        when(gigaChatApi.chatCompletionEntity(any(), any()))
                .thenReturn(new ResponseEntity<>(response, HttpStatusCode.valueOf(200)));

        gigaChatModel.call(prompt);

        ArgumentCaptor<CompletionRequest> requestCaptor = ArgumentCaptor.forClass(CompletionRequest.class);
        verify(gigaChatApi).chatCompletionEntity(requestCaptor.capture(), any());

        assertEquals(1, requestCaptor.getValue().getMessages().size());
        assertEquals(CompletionRequest.Role.system, requestCaptor.getValue().getMessages().get(0).getRole());
        assertEquals("System prompt", requestCaptor.getValue().getMessages().get(0).getContent());
    }

    @Test
    @DisplayName("AssistantMessage with tool calls is converted correctly")
    @Description("Verify that AssistantMessage with tool calls is correctly converted with function call information. " +
            "Steps: 1. Create AssistantMessage with ToolCall " +
            "2. Create prompt with UserMessage and AssistantMessage " +
            "3. Call gigaChatModel.call() " +
            "4. Verify CompletionRequest has assistant message with function call details")
    void testMessageConversion_assistantMessageWithToolCalls() {
        var toolCall = new AssistantMessage.ToolCall(
                "funcId123", "function", "testFunc", "{\"arg\":\"value\"}");
        var assistantMessage = new AssistantMessage("", Map.of(), List.of(toolCall));

        var prompt = new Prompt(List.of(new UserMessage("Test"), assistantMessage));

        when(gigaChatApi.chatCompletionEntity(any(), any()))
                .thenReturn(new ResponseEntity<>(response, HttpStatusCode.valueOf(200)));

        gigaChatModel.call(prompt);

        ArgumentCaptor<CompletionRequest> requestCaptor = ArgumentCaptor.forClass(CompletionRequest.class);
        verify(gigaChatApi).chatCompletionEntity(requestCaptor.capture(), any());

        var messages = requestCaptor.getValue().getMessages();
        var assistantMsg = messages.stream()
                .filter(m -> m.getRole() == CompletionRequest.Role.assistant)
                .findFirst()
                .orElseThrow();

        assertEquals("funcId123", assistantMsg.getFunctionsStateId());
        assertNotNull(assistantMsg.getFunctionCall());
        assertEquals("testFunc", assistantMsg.getFunctionCall().getName());
    }

    @Test
    @DisplayName("ToolResponseMessage is converted correctly")
    @Description("Verify that ToolResponseMessage is correctly converted to CompletionRequest.Message with function role. " +
            "Steps: 1. Create ToolResponseMessage with tool response " +
            "2. Create prompt with UserMessage, AssistantMessage, and ToolResponseMessage " +
            "3. Call gigaChatModel.call() " +
            "4. Verify CompletionRequest has function message with result and name")
    void testMessageConversion_toolResponseMessage() {
        var toolResponse = new ToolResponseMessage.ToolResponse("toolId", "toolName", "result");
        var toolResponseMessage = new ToolResponseMessage(List.of(toolResponse));

        var prompt = new Prompt(List.of(
                new UserMessage("Test"),
                new AssistantMessage("", Map.of(),
                        List.of(new AssistantMessage.ToolCall("toolId", "function", "toolName", "{}"))),
                toolResponseMessage));

        when(gigaChatApi.chatCompletionEntity(any(), any()))
                .thenReturn(new ResponseEntity<>(response, HttpStatusCode.valueOf(200)));

        gigaChatModel.call(prompt);

        ArgumentCaptor<CompletionRequest> requestCaptor = ArgumentCaptor.forClass(CompletionRequest.class);
        verify(gigaChatApi).chatCompletionEntity(requestCaptor.capture(), any());

        var messages = requestCaptor.getValue().getMessages();
        var functionMsg = messages.stream()
                .filter(m -> m.getRole() == CompletionRequest.Role.function)
                .findFirst()
                .orElseThrow();

        assertEquals("result", functionMsg.getContent());
        assertEquals("toolName", functionMsg.getName());
    }

    // ======================== System Prompt Sorting Tests ========================

    @Test
    @DisplayName("System prompt already first is not reordered")
    @Description("Verify that when system prompt is already first, it stays in place. " +
            "Steps: 1. Enable system prompt sorting " +
            "2. Create prompt with system message first, then user and assistant " +
            "3. Call gigaChatModel.call() " +
            "4. Verify first message in request has system role")
    void testSystemPromptSorting_whenSystemPromptFirst_shouldNotReorder() {
        when(gigaChatInternalProperties.isMakeSystemPromptFirstMessageInMemory()).thenReturn(true);

        var prompt = new Prompt(List.of(
                new SystemMessage("System"), new UserMessage("User"), new AssistantMessage("Assistant")));

        when(gigaChatApi.chatCompletionEntity(any(), any()))
                .thenReturn(new ResponseEntity<>(response, HttpStatusCode.valueOf(200)));

        gigaChatModel.call(prompt);

        ArgumentCaptor<CompletionRequest> requestCaptor = ArgumentCaptor.forClass(CompletionRequest.class);
        verify(gigaChatApi).chatCompletionEntity(requestCaptor.capture(), any());

        assertEquals(CompletionRequest.Role.system, requestCaptor.getValue().getMessages().get(0).getRole());
    }

    @Test
    @DisplayName("System prompt not first is moved to first position")
    @Description("Verify that system prompt is moved to first position when not already first. " +
            "Steps: 1. Enable system prompt sorting " +
            "2. Create prompt with user message first, then system and assistant " +
            "3. Call gigaChatModel.call() " +
            "4. Verify first message in request has system role with correct content")
    void testSystemPromptSorting_whenSystemPromptNotFirst_shouldMoveToFirst() {
        when(gigaChatInternalProperties.isMakeSystemPromptFirstMessageInMemory()).thenReturn(true);

        var prompt = new Prompt(List.of(
                new UserMessage("User"), new SystemMessage("System"), new AssistantMessage("Assistant")));

        when(gigaChatApi.chatCompletionEntity(any(), any()))
                .thenReturn(new ResponseEntity<>(response, HttpStatusCode.valueOf(200)));

        gigaChatModel.call(prompt);

        ArgumentCaptor<CompletionRequest> requestCaptor = ArgumentCaptor.forClass(CompletionRequest.class);
        verify(gigaChatApi).chatCompletionEntity(requestCaptor.capture(), any());

        assertEquals(CompletionRequest.Role.system, requestCaptor.getValue().getMessages().get(0).getRole());
        assertEquals("System", requestCaptor.getValue().getMessages().get(0).getContent());
    }

    @Test
    @DisplayName("Prompt without system message works correctly")
    @Description("Verify that prompt without system message is processed without errors. " +
            "Steps: 1. Enable system prompt sorting " +
            "2. Create prompt with only user and assistant messages " +
            "3. Call gigaChatModel.call() " +
            "4. Verify request has 2 messages without errors")
    void testSystemPromptSorting_whenNoSystemPrompt_shouldNotFail() {
        when(gigaChatInternalProperties.isMakeSystemPromptFirstMessageInMemory()).thenReturn(true);

        var prompt = new Prompt(List.of(new UserMessage("User"), new AssistantMessage("Assistant")));

        when(gigaChatApi.chatCompletionEntity(any(), any()))
                .thenReturn(new ResponseEntity<>(response, HttpStatusCode.valueOf(200)));

        gigaChatModel.call(prompt);

        ArgumentCaptor<CompletionRequest> requestCaptor = ArgumentCaptor.forClass(CompletionRequest.class);
        verify(gigaChatApi).chatCompletionEntity(requestCaptor.capture(), any());

        assertEquals(2, requestCaptor.getValue().getMessages().size());
    }

    @Test
    @DisplayName("System prompt sorting disabled preserves original order")
    @Description("Verify that when sorting is disabled, message order is preserved. " +
            "Steps: 1. Disable system prompt sorting " +
            "2. Create prompt with user message first, then system " +
            "3. Call gigaChatModel.call() " +
            "4. Verify first message in request has user role (original order)")
    void testSystemPromptSorting_whenDisabled_shouldNotSort() {
        when(gigaChatInternalProperties.isMakeSystemPromptFirstMessageInMemory()).thenReturn(false);

        var prompt = new Prompt(List.of(new UserMessage("User"), new SystemMessage("System")));

        when(gigaChatApi.chatCompletionEntity(any(), any()))
                .thenReturn(new ResponseEntity<>(response, HttpStatusCode.valueOf(200)));

        gigaChatModel.call(prompt);

        ArgumentCaptor<CompletionRequest> requestCaptor = ArgumentCaptor.forClass(CompletionRequest.class);
        verify(gigaChatApi).chatCompletionEntity(requestCaptor.capture(), any());

        assertEquals(CompletionRequest.Role.user, requestCaptor.getValue().getMessages().get(0).getRole());
    }

    // ======================== Streaming Tests ========================

    @Test
    @DisplayName("Stream with multiple chunks processes correctly")
    @Description("Verify that streaming mode correctly processes multiple response chunks. " +
            "Steps: 1. Create prompt with user message " +
            "2. Mock API to return Flux with 3 chunks (content1, content2, finish) " +
            "3. Call gigaChatModel.stream() " +
            "4. Verify all chunks are received and finish reason is 'stop'")
    void testStream_withMultipleChunks() {
        var prompt = new Prompt(List.of(new UserMessage("Hello")));

        var chunk1 = new CompletionResponse()
                .setId(UUID.randomUUID().toString())
                .setModel(GigaChatApi.ChatModel.GIGA_CHAT.getName())
                .setChoices(List.of(new CompletionResponse.Choice()
                        .setIndex(0)
                        .setDelta(new CompletionResponse.MessagesRes()
                                .setRole(CompletionResponse.Role.assistant)
                                .setContent("Hello "))));

        var chunk2 = new CompletionResponse()
                .setId(UUID.randomUUID().toString())
                .setModel(GigaChatApi.ChatModel.GIGA_CHAT.getName())
                .setChoices(List.of(new CompletionResponse.Choice()
                        .setIndex(0)
                        .setDelta(new CompletionResponse.MessagesRes().setContent("World"))));

        var chunk3 = new CompletionResponse()
                .setId(UUID.randomUUID().toString())
                .setModel(GigaChatApi.ChatModel.GIGA_CHAT.getName())
                .setChoices(List.of(new CompletionResponse.Choice()
                        .setIndex(0)
                        .setDelta(new CompletionResponse.MessagesRes().setContent(""))
                        .setFinishReason(CompletionResponse.FinishReason.STOP)));

        when(gigaChatApi.chatCompletionStream(any(), any())).thenReturn(Flux.just(chunk1, chunk2, chunk3));

        Flux<ChatResponse> chatResponseFlux = gigaChatModel.stream(prompt);

        StepVerifier.create(chatResponseFlux)
                .assertNext(response -> {
                    assertNotNull(response);
                    assertEquals(1, response.getResults().size());
                })
                .assertNext(response -> {
                    assertNotNull(response);
                })
                .assertNext(response -> {
                    assertNotNull(response);
                    assertEquals(
                            "stop",
                            response.getResults().get(0).getMetadata().getFinishReason());
                })
                .verifyComplete();
    }

    @Test
    @DisplayName("Stream with empty content handles correctly")
    @Description("Verify that streaming mode handles empty content in response correctly. " +
            "Steps: 1. Create prompt with user message " +
            "2. Mock API to return Flux with chunk containing empty content and STOP finish reason " +
            "3. Call gigaChatModel.stream() " +
            "4. Verify response contains empty text")
    void testStream_withEmptyContent() {
        var prompt = new Prompt(List.of(new UserMessage("Hello")));

        var chunk = new CompletionResponse()
                .setId(UUID.randomUUID().toString())
                .setModel(GigaChatApi.ChatModel.GIGA_CHAT.getName())
                .setChoices(List.of(new CompletionResponse.Choice()
                        .setIndex(0)
                        .setDelta(new CompletionResponse.MessagesRes()
                                .setRole(CompletionResponse.Role.assistant)
                                .setContent(""))
                        .setFinishReason(CompletionResponse.FinishReason.STOP)));

        when(gigaChatApi.chatCompletionStream(any(), any())).thenReturn(Flux.just(chunk));

        Flux<ChatResponse> chatResponseFlux = gigaChatModel.stream(prompt);

        StepVerifier.create(chatResponseFlux).assertNext(response -> {
                    assertNotNull(response);
                    assertEquals("", response.getResults().get(0).getOutput().getText());
                })
                .verifyComplete();
    }

    // ======================== Options Merging Tests ========================

    @Test
    @DisplayName("Runtime options override default options correctly")
    @Description("Verify that runtime options correctly override default options. " +
            "Steps: 1. Create GigaChatModel with default temperature 0.5 " +
            "2. Create prompt with runtime options temperature 0.8 " +
            "3. Call gigaChatModel.call() " +
            "4. Verify CompletionRequest has temperature 0.8")
    void testBuildRequestPrompt_mergeRuntimeAndDefaultOptions() {
        var defaultOptions = GigaChatOptions.builder()
                .model(GigaChatApi.ChatModel.GIGA_CHAT.getName())
                .temperature(0.5)
                .build();

        var runtimeOptions = GigaChatOptions.builder().temperature(0.8).build();

        var gigaChatModelWithDefaults = GigaChatModel.builder()
                .gigaChatApi(gigaChatApi)
                .defaultOptions(defaultOptions)
                .internalProperties(gigaChatInternalProperties)
                .build();

        var prompt = new Prompt(List.of(new UserMessage("Test")), runtimeOptions);

        when(gigaChatApi.chatCompletionEntity(any(), any()))
                .thenReturn(new ResponseEntity<>(response, HttpStatusCode.valueOf(200)));

        gigaChatModelWithDefaults.call(prompt);

        ArgumentCaptor<CompletionRequest> requestCaptor = ArgumentCaptor.forClass(CompletionRequest.class);
        verify(gigaChatApi).chatCompletionEntity(requestCaptor.capture(), any());

        assertEquals(0.8, requestCaptor.getValue().getTemperature());
    }

    @Test
    @DisplayName("Default options are used when runtime options are null")
    @Description("Verify that default options are used when no runtime options are provided. " +
            "Steps: 1. Create GigaChatModel with default temperature 0.7 " +
            "2. Create prompt without runtime options " +
            "3. Call gigaChatModel.call() " +
            "4. Verify CompletionRequest has temperature 0.7 from defaults")
    void testBuildRequestPrompt_withNullRuntimeOptions() {
        var defaultOptions = GigaChatOptions.builder()
                .model(GigaChatApi.ChatModel.GIGA_CHAT.getName())
                .temperature(0.7)
                .build();

        var gigaChatModelWithDefaults = GigaChatModel.builder()
                .gigaChatApi(gigaChatApi)
                .defaultOptions(defaultOptions)
                .internalProperties(gigaChatInternalProperties)
                .build();

        var prompt = new Prompt(List.of(new UserMessage("Test")));

        when(gigaChatApi.chatCompletionEntity(any(), any()))
                .thenReturn(new ResponseEntity<>(response, HttpStatusCode.valueOf(200)));

        gigaChatModelWithDefaults.call(prompt);

        ArgumentCaptor<CompletionRequest> requestCaptor = ArgumentCaptor.forClass(CompletionRequest.class);
        verify(gigaChatApi).chatCompletionEntity(requestCaptor.capture(), any());

        assertEquals(0.7, requestCaptor.getValue().getTemperature());
    }

    // ======================== Edge Cases Tests ========================

    @Test
    @DisplayName("Empty prompt is handled correctly")
    @Description("Verify that empty prompt (no messages) is handled without errors. " +
            "Steps: 1. Create prompt with empty message list " +
            "2. Call gigaChatModel.call() " +
            "3. Verify CompletionRequest has no messages")
    void testCall_withEmptyPrompt() {
        var prompt = new Prompt(List.of());

        when(gigaChatApi.chatCompletionEntity(any(), any()))
                .thenReturn(new ResponseEntity<>(response, HttpStatusCode.valueOf(200)));

        gigaChatModel.call(prompt);

        ArgumentCaptor<CompletionRequest> requestCaptor = ArgumentCaptor.forClass(CompletionRequest.class);
        verify(gigaChatApi).chatCompletionEntity(requestCaptor.capture(), any());

        assertEquals(0, requestCaptor.getValue().getMessages().size());
    }

    @Test
    @DisplayName("getDefaultOptions returns a copy")
    @Description("Verify that getDefaultOptions() returns a new copy each time, not the same instance. " +
            "Steps: 1. Call gigaChatModel.getDefaultOptions() twice " +
            "2. Verify returned objects are not the same instance")
    void testGetDefaultOptions_shouldReturnCopy() {
        var options1 = gigaChatModel.getDefaultOptions();
        var options2 = gigaChatModel.getDefaultOptions();

        assertNotSame(options1, options2);
    }

    @ParameterizedTest
    @MethodSource("promptAndMetadataProvider")
    @DisplayName("ChatResponse is populated with custom metadata")
    @Description("Verify that ChatResponse contains correct custom metadata including internal conversation history and uploaded media IDs. " +
            "Steps: 1. Create prompt with various message combinations " +
            "2. Call gigaChatModel.call() " +
            "3. Verify ChatResponse metadata contains expected keys and values")
    void testCustomMetadata(Prompt prompt, Map<String, Object> expectedMetadata) {
        when(gigaChatApi.chatCompletionEntity(any(), any()))
                .thenReturn(new ResponseEntity<>(response, HttpStatusCode.valueOf(200)));

        ChatResponse chatResponse = gigaChatModel.call(prompt);
        ChatResponseMetadata metadata = chatResponse.getMetadata();

        expectedMetadata.forEach((metadataKey, metadataValue) -> {
            assertEquals(metadataValue, metadata.get(metadataKey));
        });
    }

    public static Stream<Arguments> promptAndMetadataProvider() {
        return Stream.of(
                Arguments.of(
                        new Prompt(List.of(SystemMessage.builder()
                                .text("Ты - полезный ассистент")
                                .build())),
                        new HashMap<>() {
                            {
                                put(GigaChatModel.INTERNAL_CONVERSATION_HISTORY, Collections.emptyList());
                                put(GigaChatModel.UPLOADED_MEDIA_IDS, null);
                            }
                        }),
                Arguments.of(
                        new Prompt(List.of(
                                SystemMessage.builder()
                                        .text("Ты - полезный ассистент")
                                        .build(),
                                UserMessage.builder().text("Что ты умеешь?").build())),
                        new HashMap<>() {
                            {
                                put(GigaChatModel.INTERNAL_CONVERSATION_HISTORY, Collections.emptyList());
                                put(GigaChatModel.UPLOADED_MEDIA_IDS, null);
                            }
                        }),
                Arguments.of(
                        new Prompt(List.of(UserMessage.builder().text("Кто ты?").build())), new HashMap<>() {
                            {
                                put(GigaChatModel.INTERNAL_CONVERSATION_HISTORY, Collections.emptyList());
                                put(GigaChatModel.UPLOADED_MEDIA_IDS, null);
                            }
                        }),
                Arguments.of(
                        new Prompt(List.of(
                                UserMessage.builder().text("Кто ты?").build(),
                                new AssistantMessage("Я - GigaChat!"),
                                UserMessage.builder().text("Что ты умеешь?").build())),
                        new HashMap<>() {
                            {
                                put(GigaChatModel.INTERNAL_CONVERSATION_HISTORY, Collections.emptyList());
                                put(GigaChatModel.UPLOADED_MEDIA_IDS, null);
                            }
                        }),
                Arguments.of(
                        new Prompt(List.of(
                                UserMessage.builder()
                                        .text("Отправь письмо на support@chat.giga")
                                        .build(),
                                new AssistantMessage(
                                        "",
                                        Map.of(),
                                        List.of(new AssistantMessage.ToolCall(
                                                "sendEmail",
                                                "function",
                                                "sendEmail",
                                                "{\"address\": \"support@chat.giga\"}"))),
                                new ToolResponseMessage(List.of(new ToolResponseMessage.ToolResponse(
                                        "sendEmail", "sendEmail", "{\"status\": \"sent\"}"))))),
                        new HashMap<>() {
                            {
                                put(
                                        GigaChatModel.INTERNAL_CONVERSATION_HISTORY,
                                        List.of(
                                                new AssistantMessage(
                                                        "",
                                                        Map.of(),
                                                        List.of(
                                                                new AssistantMessage.ToolCall(
                                                                        "sendEmail",
                                                                        "function",
                                                                        "sendEmail",
                                                                        "{\"address\": \"support@chat.giga\"}"))),
                                                new ToolResponseMessage(List.of(new ToolResponseMessage.ToolResponse(
                                                        "sendEmail", "sendEmail", "{\"status\": \"sent\"}")))));
                                put(GigaChatModel.UPLOADED_MEDIA_IDS, null);
                            }
                        }),
                Arguments.of(
                        new Prompt(List.of(UserMessage.builder()
                                .text("Кто ты?")
                                .media(Media.builder()
                                        .id("5512e5c1-2829-4b44-ad2d-c9bce5f8b154")
                                        .data("документ")
                                        .mimeType(MimeTypeUtils.TEXT_PLAIN)
                                        .build())
                                .build())),
                        new HashMap<>() {
                            {
                                put(GigaChatModel.INTERNAL_CONVERSATION_HISTORY, Collections.emptyList());
                                put(GigaChatModel.UPLOADED_MEDIA_IDS, List.of("5512e5c1-2829-4b44-ad2d-c9bce5f8b154"));
                            }
                        }));
    }
}
