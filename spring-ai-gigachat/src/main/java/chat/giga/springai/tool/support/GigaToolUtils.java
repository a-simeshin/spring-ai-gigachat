/*
 * Copyright 2023-2025 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package chat.giga.springai.tool.support;

import chat.giga.springai.tool.annotation.FewShotExample;
import chat.giga.springai.tool.annotation.FewShotExampleList;
import chat.giga.springai.tool.annotation.GigaTool;
import chat.giga.springai.tool.execution.GigaToolCallResultConverter;
import com.fasterxml.jackson.core.JacksonException;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.databind.json.JsonMapper;
import java.lang.reflect.Method;
import java.lang.reflect.Type;
import java.util.Arrays;
import java.util.Set;
import java.util.stream.Stream;
import lombok.experimental.UtilityClass;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.tool.annotation.Tool;
import org.springframework.ai.tool.execution.ToolCallResultConverter;
import org.springframework.ai.util.JacksonUtils;
import org.springframework.ai.util.ParsingUtils;
import org.springframework.ai.util.json.schema.JsonSchemaGenerator;
import org.springframework.lang.Nullable;
import org.springframework.util.Assert;
import org.springframework.util.ClassUtils;
import org.springframework.util.StringUtils;

/**
 * Utility class providing support functionality for integrating Tool Calling API with GigaChat LLM.
 * <p>
 * <strong>Main Features:</strong>
 * <ul>
 *     <li>Extracting tool metadata from annotations</li>
 *     <li>Generating JSON schemas for tool parameters and responses</li>
 *     <li>Handling few-shot examples for prompt engineering</li>
 *     <li>JSON serialization/deserialization utilities</li>
 * </ul>
 *
 * @author Linar Abzaltdinov
 */
@UtilityClass
@Slf4j
public final class GigaToolUtils {

    /**
     * Configured ObjectMapper instance used throughout the tool support package.
     * <ul>
     *     <li>FAIL_ON_TRAILING_TOKENS enabled for strict deserialization</li>
     *     <li>FAIL_ON_UNKNOWN_PROPERTIES disabled for flexible deserialization</li>
     *     <li>FAIL_ON_EMPTY_BEANS disabled for empty object handling</li>
     *     <li>Includes all available Jackson modules</li>
     * </ul>
     */
    private static final ObjectMapper OBJECT_MAPPER = JsonMapper.builder()
            .enable(DeserializationFeature.FAIL_ON_TRAILING_TOKENS)
            .disable(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES)
            .disable(SerializationFeature.FAIL_ON_EMPTY_BEANS)
            .addModules(JacksonUtils.instantiateAvailableModules())
            .build();

    /**
     * Error message constant used for null method validation.
     * @see Assert#notNull(Object, String)
     */
    private static final String ASSERT_METHOD = "method cannot be null";

    /**
     * Set of Java primitive wrapper types that don't require JSON schema generation.
     * <ul>
     *     <li>String.class</li>
     *     <li>Byte.class</li>
     *     <li>Integer.class</li>
     *     <li>Short.class</li>
     *     <li>Long.class</li>
     *     <li>Double.class</li>
     *     <li>Float.class</li>
     *     <li>Boolean.class</li>
     * </ul>
     * @see #generateJsonSchemaForOutputType(Type)
     */
    private static final Set<Class<?>> JAVA_LANG_WRAPPERS = Set.of(
            String.class, Byte.class, Integer.class, Short.class, Long.class, Double.class, Float.class, Boolean.class);

    /**
     * Retrieves the effective tool name from method annotations.
     * <ul>
     *     <li>First checks for GigaTool.name()</li>
     *     <li>Then checks Tool.name()</li>
     *     <li>Falls back to method name</li>
     * </ul>
     *
     * @param method The method to inspect for annotations
     * @return The resolved tool name
     * @throws IllegalArgumentException if method is null
     */
    public static String getToolName(final Method method) {
        Assert.notNull(method, ASSERT_METHOD);
        final GigaTool gigaTool = method.getAnnotation(GigaTool.class);
        if (gigaTool != null && StringUtils.hasText(gigaTool.name())) {
            return gigaTool.name();
        }
        final Tool tool = method.getAnnotation(Tool.class);
        if (tool != null && StringUtils.hasText(tool.name())) {
            return tool.name();
        }
        return method.getName();
    }

    /**
     * Retrieves the tool description from method annotations.
     * <ul>
     *     <li>Prioritizes GigaTool.description()</li>
     *     <li>Then checks Tool.description()</li>
     *     <li>Finally converts camelCase method name to readable text</li>
     * </ul>
     *
     * @param method The method to inspect
     * @return The resolved tool description
     * @throws IllegalArgumentException if method is null
     */
    public static String getToolDescription(final Method method) {
        Assert.notNull(method, ASSERT_METHOD);
        final GigaTool gigaTool = method.getAnnotation(GigaTool.class);
        if (gigaTool != null && StringUtils.hasText(gigaTool.description())) {
            return gigaTool.description();
        }
        final Tool tool = method.getAnnotation(Tool.class);
        if (tool != null && StringUtils.hasText(tool.description())) {
            return tool.description();
        }
        return ParsingUtils.reConcatenateCamelCase(method.getName(), " ");
    }

    /**
     * Determines if the tool should return results directly.
     * <ul>
     *     <li>Checks GigaTool.returnDirect() first</li>
     *     <li>Then checks Tool.returnDirect()</li>
     * </ul>
     *
     * @param method The method to check
     * @return true if results should be returned directly
     * @throws IllegalArgumentException if method is null
     */
    public static boolean getToolReturnDirect(final Method method) {
        Assert.notNull(method, ASSERT_METHOD);
        final GigaTool gigaTool = method.getAnnotation(GigaTool.class);
        if (gigaTool != null) {
            return gigaTool.returnDirect();
        }
        final Tool tool = method.getAnnotation(Tool.class);
        return tool != null && tool.returnDirect();
    }

    /**
     * Creates a result converter instance based on method annotations.
     * <ul>
     *     <li>Uses GigaTool.resultConverter() if available</li>
     *     <li>Otherwise uses Tool.resultConverter()</li>
     *     <li>Defaults to GigaToolCallResultConverter if neither annotation exists</li>
     * </ul>
     *
     * @param method The method to get converter for
     * @return A new ToolCallResultConverter instance
     * @throws IllegalArgumentException if converter instantiation fails
     * @throws IllegalArgumentException if method is null
     */
    public static ToolCallResultConverter getToolCallResultConverter(final Method method) {
        Assert.notNull(method, ASSERT_METHOD);
        final Tool tool = method.getAnnotation(Tool.class);
        final GigaTool gigaTool = method.getAnnotation(GigaTool.class);
        if (tool == null && gigaTool == null) {
            return new GigaToolCallResultConverter();
        }
        final Class<? extends ToolCallResultConverter> type =
                gigaTool != null ? gigaTool.resultConverter() : tool.resultConverter();
        try {
            return type.getDeclaredConstructor().newInstance();
        } catch (Exception e) {
            throw new IllegalArgumentException("Failed to instantiate ToolCallResultConverter: " + type, e);
        }
    }

    /**
     * Aggregates all available few-shot examples from method annotations.
     * <ul>
     *     <li>Combines examples from individual @FewShotExample annotations</li>
     *     <li>Includes examples from @FewShotExampleList collections</li>
     *     <li>Includes examples from GigaTool.fewShotExamples()</li>
     * </ul>
     *
     * @param method The method to extract examples from
     * @return Array of few-shot examples (may be empty)
     * @throws IllegalArgumentException if method is null
     */
    public static chat.giga.springai.tool.definition.FewShotExample[] getFewShotExamples(final Method method) {
        Assert.notNull(method, ASSERT_METHOD);

        final GigaTool gigaTool = method.getAnnotation(GigaTool.class);
        final Tool tool = method.getAnnotation(Tool.class);
        if (gigaTool == null && tool == null) {
            return new chat.giga.springai.tool.definition.FewShotExample[0];
        }

        final FewShotExampleList exampleList = method.getAnnotation(FewShotExampleList.class);
        final FewShotExample example = method.getAnnotation(FewShotExample.class);
        Stream<FewShotExample> shotExampleStream = Stream.empty();
        if (example != null) {
            shotExampleStream = Stream.of(example);
        }
        if (exampleList != null && exampleList.value() != null) {
            shotExampleStream = Stream.concat(shotExampleStream, Arrays.stream(exampleList.value()));
        }
        if (gigaTool != null && gigaTool.fewShotExamples() != null) {
            shotExampleStream = Stream.concat(shotExampleStream, Arrays.stream(gigaTool.fewShotExamples()));
        }

        return shotExampleStream
                .map(fewShotExample -> chat.giga.springai.tool.definition.FewShotExample.builder()
                        .request(fewShotExample.request())
                        .paramsSchema(fewShotExample.params())
                        .build())
                .toArray(chat.giga.springai.tool.definition.FewShotExample[]::new);
    }

    /**
     * Generates JSON schema for a method's return type if configured.
     * <ul>
     *     <li>Only generates schema if GigaTool.generateOutputSchema() is true</li>
     *     <li>Skips generation for primitive wrappers, enums, or arrays</li>
     * </ul>
     *
     * @param method The method to generate schema for
     * @return JSON schema string or null
     * @throws IllegalArgumentException if method is null
     */
    public static String generateJsonSchemaForMethodOutput(final Method method) {
        Assert.notNull(method, ASSERT_METHOD);
        final GigaTool gigaTool = method.getAnnotation(GigaTool.class);
        if (gigaTool == null || !gigaTool.generateOutputSchema()) {
            return null;
        }
        final Class<?> returnType = method.getReturnType();
        return generateJsonSchemaForOutputType(returnType);
    }

    /**
     * Generates JSON schema for any type, skipping simple types.
     * <ul>
     *     <li>Returns null for String types</li>
     *     <li>Returns null for Enum types</li>
     *     <li>Returns null for primitive wrappers</li>
     *     <li>Returns null for array types</li>
     * </ul>
     *
     * @param type The type to generate schema for
     * @return JSON schema string or null
     */
    public static String generateJsonSchemaForOutputType(final Type type) {
        if (type instanceof Class<?> clazz) {
            final Class<?> javaType = ClassUtils.resolvePrimitiveIfNecessary(clazz);
            if (JAVA_LANG_WRAPPERS.contains(javaType) || javaType.isEnum() || javaType.isArray()) {
                log.info("Skipping schema generation for primitive or array type: {}", type);
                return null;
            }
        }
        return JsonSchemaGenerator.generateForType(type);
    }

    /**
     * Checks if a given JSON string is valid.
     *
     * @param json The JSON string to validate
     * @return true if valid JSON, false otherwise
     * @throws IllegalArgumentException if json is null
     */
    public static boolean isValidJson(final String json) {
        Assert.notNull(json, "json cannot be null");
        try {
            OBJECT_MAPPER.readTree(json);
        } catch (JacksonException e) {
            return false;
        }
        return true;
    }

    /**
     * Converts an object to its JSON string representation.
     *
     * @param object The object to convert (maybe null)
     * @return JSON string representation
     * @throws IllegalStateException if conversion fails
     */
    public static String toJson(@Nullable final Object object) {
        try {
            return OBJECT_MAPPER.writeValueAsString(object);
        } catch (JacksonException ex) {
            throw new IllegalStateException("Conversion from Object to JSON failed", ex);
        }
    }

    /**
     * Converts an object to JSON only if it's not already valid JSON.
     *
     * @param param The parameter to convert
     * @return JSON string representation
     * @throws IllegalStateException if JSON conversion fails
     */
    public static String toJsonIfNeeded(final Object param) {
        if (param instanceof String paramStr && isValidJson(paramStr)) {
            return paramStr;
        } else {
            return toJson(param);
        }
    }
}
