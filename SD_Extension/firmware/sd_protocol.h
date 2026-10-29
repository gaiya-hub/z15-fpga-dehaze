#pragma once
#include <stdint.h>
#define SD_INPUT_MAGIC 0x31495a44U
#define SD_INDEX_MAGIC 0x31585a44U
#define SD_RESULT_MAGIC 0x31525a44U
#define SD_BATCH_MAGIC 0x31425a44U
#define SD_VERSION 1U
#define SD_RESULT_WORDS 128U
#define SD_MOUNT_ERROR 0x53440001U
#define SD_TASK_ERROR 0x53440002U
#define SD_JOB_ERROR 0x53440003U
#define SD_BATCH_DONE 0x53444f4bU
#ifndef SD_BUILD_ID
#define SD_BUILD_ID 0U
#endif
static uint32_t sd_crc_table[256];
static void sd_crc_init(void){for(unsigned i=0;i<256;i++){uint32_t c=i;for(int j=0;j<8;j++)c=(c>>1)^((c&1)?0xedb88320U:0);sd_crc_table[i]=c;}}
static uint32_t sd_crc(const void *data,unsigned n){const unsigned char *p=data;uint32_t c=~0U;for(unsigned i=0;i<n;i++)c=sd_crc_table[(c^p[i])&255]^(c>>8);return ~c;}
