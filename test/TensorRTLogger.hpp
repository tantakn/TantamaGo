/**
 * @file include/dl/trt/TensorRTLogger.hpp
 * @author Yuki Kobayashi
 * @~english
 * @brief Definition of Logger for TensorRT.
 * @~japanese
 * @brief TensorRT用ロガーの定義
 */
#ifndef _TENSOR_RT_LOGGER_HPP_
#define _TENSOR_RT_LOGGER_HPP_

#if defined USE_TENSOR_RT

#include <iostream>

#include "NvInferRuntimeCommon.h"


/**
 * @class TensorRTLogger
 * @~english
 * @brief Logger for TensorRT.
 * @~japanese
 * @brief TensorRT用のロガー
 */
class TensorRTLogger : public nvinfer1::ILogger {
    /**
     * @~english
     * @brief Get severity type string.
     * @param[in] severity Severity.
     * @return Severity type string.
     * @~japanese
     * @brief 重大度文字列の取得
     * @param[in] severity 重大度
     * @return 重大度文字列
     */
    const char*
    get_error_type(const Severity severity)
    {
        switch (severity) {
            case Severity::kINTERNAL_ERROR:
                return "[FATAL] ";
            case Severity::kERROR:
                return "[ERROR] ";
            case Severity::kWARNING:
                return "[WARNING] ";
            case Severity::kINFO:
                return "[INFO] ";
            case Severity::kVERBOSE:
                return "[VERBOSE] ";
            default:
                return "";
        }
    }

    /**
     * @~english
     * @brief Output log message.
     * @param[in] severity Severity.
     * @param[in] message Log message.
     * @~japanese
     * @brief ログメッセージのコンソール出力
     * @param[in] severity 重大度
     * @param[in] message ログメッセージ
     */
    void
    log(const Severity severity, const char* message) noexcept
    {
        if (severity <= Severity::kERROR) {
            std::cerr << get_error_type(severity) << message << std::endl;
        }
    }
};

#endif

#endif