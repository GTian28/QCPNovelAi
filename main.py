# -*- coding: utf-8 -*-
"""
@CreateTime     : 2023/1/30 12:17
@Author         : DominoAR and group(member_name)
@File           : main.py.py
@LastEditTime   : 
"""
import asyncio
import hashlib
import json
import os
import re
import sqlite3
import time
import traceback
from logging import Logger, StreamHandler
from typing import Optional, List

import mirai
import requests
import yaml
from aiohttp import ClientSession

from pkg.plugin.host import PluginHost, EventContext
from pkg.plugin.models import *
from plugins.NovelAi.novelai_api.BanList import BanList
from plugins.NovelAi.novelai_api.BiasGroup import BiasGroup
from plugins.NovelAi.novelai_api.GlobalSettings import GlobalSettings
from plugins.NovelAi.novelai_api.ImagePreset import ImageModel, ImagePreset, ImageResolution, ImageSampler, UCPreset
from plugins.NovelAi.novelai_api.NovelAI_API import NovelAIAPI
from plugins.NovelAi.novelai_api.Preset import Model, Preset
from plugins.NovelAi.novelai_api.Tokenizer import Tokenizer
from plugins.NovelAi.novelai_api.utils import get_encryption_key, b64_to_tokens


class API:
    _username: str
    _password: str
    _session: ClientSession

    logger: Logger
    api: Optional[NovelAIAPI]

    def __init__(self):
        config_dict = yaml.load(open(f'{os.getcwd()}/plugins/NovelAi/config.yaml', mode='r', encoding='utf-8').read(),
                                yaml.CLoader)

        self._username = config_dict['account']['user']
        self._password = config_dict['account']['password']
        if self._username is None or self._password is None:
            raise RuntimeError("NovelAi：账号或密码错误，账号或密码不能为空，且必须为字符串")
        if self._username == "" or self._password == "":
            raise RuntimeError("NovelAi：账号或密码错误，账号或密码不能为空")

        self.logger = Logger("NovelAI")
        self.logger.addHandler(StreamHandler())

        self.api = NovelAIAPI(logger=self.logger)

    @property
    def encryption_key(self):
        return get_encryption_key(self._username, self._password)

    async def __aenter__(self):
        self._session = ClientSession()
        await self._session.__aenter__()

        self.api.attach_session(self._session)
        await self.api.high_level.login(self._username, self._password)

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._session.__aexit__(exc_type, exc_val, exc_tb)


class NovelAiStory:
    """NovelAi故事插件的功能类"""
    # 故事会话池
    story_pool = {}
    username: str
    password: str
    mode: str
    text_length: int
    banlist: list

    def __init__(self):
        # 读取配置文件
        logging.debug("NovelAi：开始读取配置文件")
        try:
            this_path = os.getcwd()
            config_dict = yaml.load(open(f'{this_path}/plugins/NovelAi/config.yaml', mode='r', encoding='utf-8').read(),
                                    yaml.CLoader)
            self.username = config_dict['account']['user']
            self.password = config_dict['account']['password']
            self.mode = config_dict['story']['mode']
            self.text_length = config_dict['story']['text_length']
            self.banlist = config_dict['story']['banlist']
        except Exception:
            traceback.print_exc()
            raise RuntimeError("NovelAi：读取文件失败，请检查配置文件是否正确")

    async def process_mod(self, massage):
        """入口函数"""
        reply = await self.get_novel_reply(massage)
        return reply

    async def get_novel_reply(self, massage):
        """获取novelai的回复"""
        async with API() as api_handler:
            api = api_handler.api
            logger = api_handler.logger
            if self.mode == "Euterpe":
                model = Model.Euterpe
            elif self.mode == "Sigurd":
                model = Model.Sigurd
            elif self.mode == "Krake":
                model = Model.Krake
            prompt = massage
            preset = Preset.from_default(model)
            preset["max_length"] = self.text_length
            global_settings = GlobalSettings(num_logprobs=GlobalSettings.NO_LOGPROBS)
            global_settings["bias_dinkus_asterism"] = True
            ban_words = BanList()
            for word in self.banlist:
                ban_words.add(word)
            bad_words: Optional[BanList] = ban_words
            bias_groups: List[BiasGroup] = []
            bias_group1 = BiasGroup(0.00)
            bias_group2 = BiasGroup(0.00)
            if bias_groups:
                bias_group1.add("very", " very", " slightly", " incredibly", " enormously", " loudly")
                bias_group1 += " proverbially"
                bias_group2 += " interestingly"
                bias_group2 += " brutally"
            module = None
            gen = await api.high_level.generate(prompt, model, preset, global_settings, bad_words, bias_groups,
                                                module)
            reply_msg = Tokenizer.decode(model, b64_to_tokens(gen["output"]))
            return reply_msg


def get_novel_model(cmd: str, novel_config: dict) -> dict:
    """获取参数的novelai模型
    :param cmd:模型缩写
    :param novel_config:配置文件
    :return: dict{"novel_model":模型}
    """
    model = NovelAiImage.novel_model_dict.get(cmd)
    if model:
        return {"novel_model": model}
    return {"novel_model": ImageModel.Anime_Curated}


def get_novel_size(cmd: str, novel_config: dict) -> dict:
    """获取参数的novelai图片像素大小
    :param cmd:模型预设像素缩写
    :param novel_config:配置文件
    :return: dict{"novel_size":预设大小}
    """
    # 预设大小或者是指定了大小
    if re.search("\dx\d", cmd):
        size = re.findall("(\d+)x(\d+)", cmd)[0]
    else:
        size = NovelAiImage.novel_model_dict.get(cmd)
    if size:
        return {"novel_size": size}
    return {"novel_size": ImageResolution.Large_Landscape}


def get_novel_bad_tag(tag: str, novel_config: dict) -> dict:
    """获取参数的负面标签
    :param tag:负面标签
    :param novel_config:配置文件
    :return: {"novel_bad_tag":标签}
    """
    return {"novel_bad_tag": "{}{}".format(novel_config.get("image").get("default_reverse_tag"), tag)}


def get_novel_sampler(cmd: str, novel_config) -> dict:
    """获取参数的采集器
    :param cmd:采集器名称缩写
    :param novel_config:配置文件
    :return: {"novel_sampler":采集器}
    """
    sampler = NovelAiImage.novel_sampler.get(cmd)
    if sampler is None:
        sampler = ImageSampler.k_euler_ancestral
    return {"novel_sampler": sampler}


def get_novel_seed(cmd: str, novel_config) -> dict:
    """获取参数的随机种子
    :param cmd:字符类型的数字
    :param novel_config:配置文件
    :return: {"novel_seed",数字}
    """
    return {"novel_seed": int(cmd)}


def get_novel_noise(cmd: str, novel_config) -> dict:
    """获取参数的噪音
    :param cmd:字符类型的数字
    :param novel_config:配置文件
    :return:{"novel_noise":数字}
    """
    return {"novel_noise": int(cmd)}


def get_novel_strength(cmd: str, novel_config) -> dict:
    """获取参数的强度
    :param cmd:字符类型的数字
    :param novel_config:配置文件
    :return: {"novel_strength":数字}
    """
    return {"novel_strength": int(cmd)}


def get_novel_steps(cmd: str, novel_config) -> dict:
    """获取参数的迭代步数
    :param cmd:字符类型的数字
    :param novel_config:配置文件
    :return:{"novel_steps":数字}
    """
    return {"novel_steps": int(cmd)}


def get_novel_scale(cmd: str, novel_config) -> dict:
    """获取参数的服从度
    :param cmd: 字符类型的数字
    :param novel_config:配置文件
    :return:{"novel_scale":数字}
    """
    return {"novel_scale": int(cmd)}


command_dict = {"-m": get_novel_model, "-r": get_novel_size, "-s": get_novel_sampler,
                "-x": get_novel_seed, "-t": get_novel_steps, "-c": get_novel_scale, "-N": get_novel_strength,
                "-n": get_novel_noise, "-u": get_novel_bad_tag, "negative prompt": get_novel_bad_tag,
                "--model": get_novel_model, "--resolution": get_novel_size, "--sampler": get_novel_sampler,
                "--seed": get_novel_seed, "--steps": get_novel_steps, "--scale": get_novel_scale,
                "--strength": get_novel_strength, "--noise": get_novel_noise}


class NovelAiImage:
    """NovelAi图片类"""
    username: str
    password: str
    img_save_path: str
    novel_model_dict = {
        "sd": ImageModel.Anime_Curated,
        "nd": ImageModel.Anime_Full,
        "ndf": ImageModel.Furry
    }
    novel_size_dict = {
        "sp": ImageResolution.Small_Portrait,
        "sl": ImageResolution.Small_Landscape,
        "ss": ImageResolution.Small_Square,
        "np": ImageResolution.Normal_Portrait,
        "nl": ImageResolution.Normal_Landscape,
        "ns": ImageResolution.Normal_Square,
        "lp": ImageResolution.Large_Portrait,
        "ll": ImageResolution.Large_Landscape,
        "ls": ImageResolution.Large_Square
    }
    novel_sampler = {
        "kl": ImageSampler.k_lms,
        "ke": ImageSampler.k_euler,
        "kea": ImageSampler.k_euler_ancestral,
        "p": ImageSampler.plms,
        "d": ImageSampler.ddim
    }

    def __init__(self):
        logging.debug("[NovelAi]: 正在读取配置文件……")
        # 创建图片保存路径
        this_file_path = os.getcwd()
        self.img_save_path = "{}/plugins/NovelAi/novel-image".format(this_file_path)
        try:
            os.mkdir(self.img_save_path)
        except FileExistsError:
            pass
        # 读取配置文件
        with open(this_file_path + '/plugins/NovelAi/config.yaml', 'r', encoding='utf-8') as conf_yaml:
            config_yaml = yaml.load(conf_yaml.read(), yaml.CLoader)
            self.username = config_yaml.get("account").get("user")
            self.password = config_yaml.get("account").get("password")
        logging.debug("[NovelAi]: 读取配置文件完成！")

    async def process_mod(self, tag: str, param_list: list, sender_id: int, novel_config: dict) -> None:
        """插件入口
        :param tag: 绘画标签
        :param param_list: 绘画指令列表
        :param sender_id: 指令发送者
        :param novel_config: 配置文件
        :return: None
        """
        global command_dict
        novel_params = {"sender_id": sender_id}
        # 执行指令
        for param in param_list:
            novel_params.update(command_dict[param[0]](param[1], novel_config))
        novel_params.update({"novel_tag": tag})
        await self._generate_image(**novel_params)

    @staticmethod
    async def _generate_image(*args, **kwargs) -> None:
        """创建一个新的图片，(参数参见novelai-api/ImagePreset.py)
        :param kwargs:参数字典
        :return:在本地生成图片，无返回值
        """

        async with API() as api_handler:
            api = api_handler.api
            preset = ImagePreset()

            # 加载参数设置
            novel_tag = kwargs.get("novel_tag")
            novel_bad_tag = kwargs.get("novel_bad_tag")
            novel_model = kwargs.get("novel_model")
            novel_size = kwargs.get("novel_size")
            novel_sampler = kwargs.get("novel_sampler")
            novel_seed = kwargs.get("novel_seed")
            novel_steps = kwargs.get("novel_steps")
            novel_scale = kwargs.get("novel_scale")
            novel_strength = kwargs.get("novel_strength")
            novel_noise = kwargs.get("novel_noise")

            if novel_model is None:
                novel_model = ImageModel.Anime_Curated
            if novel_size:
                preset["resolution"] = novel_size
            if novel_sampler:
                preset["sampler"] = novel_sampler
            else:
                preset["sampler"] = ImageSampler.k_euler_ancestral
            if novel_seed:
                preset["seed"] = novel_seed
            if novel_steps:
                preset["steps"] = novel_steps
            if novel_scale:
                preset["scale"] = novel_scale
            if novel_strength:
                preset["strength"] = novel_strength
            if novel_noise:
                preset["noise"] = novel_noise
            if novel_bad_tag:
                preset.set_custom_uc_preset(novel_model, novel_bad_tag)
                preset["uc_preset"] = UCPreset.Preset_Custom

            # hash md5不唯一生成图片 Tag加上发送者QQ号码
            sender_id = kwargs.get("sender_id")
            file_md5 = hashlib.md5()
            file_md5.update("{}{}".format(str(sender_id), novel_tag).encode("utf-8"))
            img_md5 = file_md5.hexdigest()

            async for img in api.high_level.generate_image(novel_tag, novel_model, preset):
                with open("novelai-{}-img.png".format(img_md5), "wb") as f:
                    f.write(img)


# 记录会话novelai story的开启状态
# 存在号码即是开启，反之关闭
novel_status = []


@register(name="欢迎使用基于QChatGPT的NovelAi插件(*^▽^*)", description="#", version="v0.1.2",
          author="多米诺艾尔(Dominoar　&　ドミノァイエ!)")
class NovalAiStoryPlugins(Plugin):
    sqlite = None
    cursor = None

    def __init__(self, plugin_host: PluginHost):
        asyncio.set_event_loop(asyncio.new_event_loop())
        # 加载配置文件
        with open(f'{os.getcwd()}/plugins/NovelAi/config.yaml', mode='r', encoding='UTF-8') as novel_conf_file:
            self.novel_config = yaml.load(novel_conf_file.read(), yaml.CLoader)
        # 创建novel故事Ai
        self.novel_story = NovelAiStory()
        # 创建异步loop
        self.async_loop = asyncio.get_event_loop()
        # 初始化数据库
        self.sqlite = sqlite3.connect(f"{os.getcwd()}/database.db")
        self.cursor = self.sqlite.cursor()
        create_tb_sql = '''
                            CREATE TABLE IF NOT EXISTS novel_content (
                            person_id int,
                            type varchar(10),
                            datatime integer,
                            content varchar(8000));
                            '''
        self.cursor.execute(create_tb_sql)
        self.sqlite.commit()
        self.cursor.close()
        self.sqlite.close()

    @on(PersonNormalMessageReceived)
    @on(GroupNormalMessageReceived)
    def normal_message_received(self, event: EventContext, **kwargs):
        person_id = kwargs['launcher_id']
        person_msg = kwargs['text_message']
        # 是否覆盖draw指令
        draw_cmd = "^绘画\x20|^draw\x20"
        # 如果该人(群)开启了NovelAi故事模式
        if person_id in novel_status:
            # 获取接下来需要使用到的各个参数
            launcher_type = kwargs['launcher_type']
            language = self.novel_config.get('story').get('language')
            trans_choice = self.novel_config.get("Translate").get("your_choice")
            # 数据库
            self.sqlite = sqlite3.connect(f"{os.getcwd()}/database.db")
            self.cursor = self.sqlite.cursor()
            # 翻译用户输入的中文文本消息 -> 英文
            if language == 'zh':
                en_trans_msg = translate_chinese_check(trans_choice, person_msg, 1, novel_config=self.novel_config)
            else:
                en_trans_msg = person_msg
            # 获取用户content文本
            person_content = self._get_db_contents(person_id, en_trans_msg, launcher_type)
            # 开始获取NovelAi回复(异步)
            novel_task = self.async_loop.create_task(self.novel_story.process_mod(person_content))
            self.async_loop.run_until_complete(novel_task)
            reply = novel_task.result()
            # 将回复写入数据库
            self._set_db_content(person_id, context=person_content + reply)
            self._delete_db_timeout()
            # 翻译NovelAi的回复
            if language == 'zh':
                zh_trans_reply = translate_chinese_check(trans_choice, reply, 0, novel_config=self.novel_config)
                event.add_return("reply", [zh_trans_reply])
            else:
                event.add_return("reply", reply)
            event.prevent_default()
        # NovelAi图片绘画
        if re.match(draw_cmd, person_msg):
            host_event: PluginHost = kwargs.get("host")
            sender_id = kwargs.get("sender_id")
            launcher_id = kwargs.get("launcher_id")
            launcher_type = kwargs.get("launcher_type")
            # 判断是否是帮助信息
            if re.match("^绘画$|^绘.*?[助p]$|^d.*?[助p]$", person_msg):
                help_msg = """欢迎使用QChatGPT NovelAi插件
用法：
1、使用draw或者绘画来进行绘画
2、更多使用方法{参考}此处：
https://bot.novelai.dev/usage.html
3、学习NovelAi元素宝典！
https://docs.qq.com/doc/DWHl3am5Zb05QbGVs
如果你喜欢本插件，欢迎 star ~
https://github.com/dominoar/QCP-NovelAi"""
                event.add_return("reply", help_msg)
            else:
                # 处理指令
                person_msg += "\x20"
                lite_cmds = re.findall(r"(-.)\x20(.*?)\x20", person_msg)
                long_cmds = re.findall(r"(--.*?)\x20(.*?)\x20", person_msg)
                try:
                    bad_tag = re.findall(r"(negative prompt):\x20(.*?)\x20", person_msg)[0]
                except IndexError:
                    bad_tag = ("-u", None)
                tag = re.sub(
                    "{}{}".format(draw_cmd,
                                  "|(-.)\x20(.*?)\x20|(--.*?)\x20(.*?)\x20|(negative\x20prompt):\x20(.*?)\x20"),
                    "",
                    person_msg)
                # 提纯
                tag = re.sub("^\x20+|\x20+$", "", tag)
                for cc in long_cmds:
                    lite_cmds.append(cc)
                lite_cmds.append(bad_tag)
                asyncio.run(NovelAiImage().process_mod(tag, lite_cmds, sender_id, novel_config=self.novel_config))
                # md5读取图片并发送
                hash_md5 = hashlib.md5()
                hash_md5.update("{}{}".format(str(sender_id), tag).encode("utf-8"))
                img_md5 = hash_md5.hexdigest()
                mirai_img = mirai.Image(path="novelai-{}-img.png".format(img_md5))
                if launcher_type == "group":
                    host_event.send_group_message(launcher_id, mirai_img)
                else:
                    host_event.send_person_message(launcher_id, mirai_img)
                os.remove("novelai-{}-img.png".format(img_md5))
                event.prevent_default()

    @on(GroupCommandSent)
    @on(PersonCommandSent)
    def command_sent_event(self, event: EventContext, **kwargs):
        cmd: str = kwargs.get('command')
        params_list: list = kwargs.get('params')
        # 查询cmd指令
        if re.search('story|故事', cmd):
            self.sqlite = sqlite3.connect(f"{os.getcwd()}/database.db")
            self.cursor = self.sqlite.cursor()
            if_admin = not kwargs.get('is_admin')
            launcher_id = kwargs.get('launcher_id')
            # 指令列表必须大于0
            if len(params_list) > 0:
                if re.search('^reset|^重置会话', params_list[0]):
                    if len(params_list) == 1:
                        self._reset_one_db_session(launcher_id=launcher_id)
                        event.add_return("reply", ["已完成重置个人会话"])
                    elif len(params_list) == 2 and re.match('^all|^全部', params_list[1]) and if_admin:
                        self._reset_all_db_session()
                        event.add_return("reply", ["已完成重置所有会话"])
                    else:
                        if not if_admin:
                            event.add_return("reply", ["哼~ 这个功能只有管理员才能用你不知道吗？(ｰ̀дｰ́)"])
                        else:
                            event.add_return("reply", ["笨蛋！参数错啦,这一级指令只有[all|全部]可用！"])
                elif re.match('^start|^启|^开', params_list[0]):
                    if novel_status.count(launcher_id):
                        event.add_return("reply", ["你已经进入故事模式~ ヾ(✿ﾟ▽ﾟ)ノ"])
                    else:
                        novel_status.append(launcher_id)
                        event.add_return("reply", ["成功进入故事模式~ ヾ(✿ﾟ▽ﾟ)ノ"])
                elif re.match('^stop|^停|^退|^关', params_list[0]):
                    if novel_status.count(launcher_id) == 0:
                        event.add_return("reply", ["你从未进入故事模式"])
                    else:
                        novel_status.remove(launcher_id)
                        event.add_return("reply", ["成功退出故事模式 |ू･ω･` )"])
                else:
                    event.add_return("reply", ["你是不是傻(￣.￣),这一级指令只有[reset|start|stop]可用哦~"])
            else:
                event.add_return("reply", ["用法：!story [reset(重置会话) start(开启故事模式) stop(关闭故事模式)]"])
            event.prevent_default()

    # 下面是数据库操作
    def _set_db_content(self, person_id, context):
        if len(context) > 8000:  # context不能超过8000个字符,否则重置到4000
            context = context[4000:]
        if re.search(r'\.$', context) is None:
            context += '.'
        context = re.sub('\n', '.', context)
        # 字符串sql转义单，双引号
        context = re.sub('"', '“', context)
        context = re.sub("'", '”', context)
        sql = """
            UPDATE novel_content
            SET content = '%s' , datatime = %s
            WHERE person_id = '%s';
            """ % (context, int(time.time()), person_id)
        self.cursor.execute(sql)
        self.sqlite.commit()

    def _get_db_contents(self, person_id, person_msg, launcher_type):
        """
        获取目标成员的文本,如果没有该成员则创建一个成员
        :param person_id: 消息发送者
        :param person_msg: 消息内容
        :param launcher_type: 消息源自于(私聊/群组)
        """
        select_sql = """
            SELECT content FROM novel_content WHERE person_id = %s;
            """ % person_id
        contents = self.cursor.execute(select_sql)
        context = contents.fetchone()
        if context:
            context = context[0]
            context = re.sub('“', '"', context)
            context = re.sub("”", "'", context)
            return context + person_msg
        else:  # 如果是第一次会话
            person_msg = re.sub('"', '“', person_msg)
            person_msg = re.sub("'", '”', person_msg)
            crete_sql = """
            INSERT INTO novel_content (person_id, content,type,datatime) VALUES (%s,'%s','%s',%s)
            """ % (person_id, person_msg, launcher_type, int(time.time()))
            self.cursor.execute(crete_sql)
            self.sqlite.commit()
            return person_msg

    def _delete_db_timeout(self):
        """删除超过20分钟的数据"""
        delete_sql = """
            DELETE FROM novel_content WHERE datatime < %s
            """ % (int(time.time()) - 1200)
        self.cursor.execute(delete_sql)
        self.sqlite.commit()

    def _reset_all_db_session(self):
        """删除所有数据库会话"""
        delete_sql = """
            DELETE FROM novel_content WHERE TRUE
            """
        self.cursor.execute(delete_sql)
        self.sqlite.commit()

    def _reset_one_db_session(self, launcher_id):
        """
        删除某一个人(群聊)的会话
        :param launcher_id: 发送消息的人(群)的号码
        """
        delete_sql = """
            DELETE FROM novel_content WHERE person_id = %s
            """ % launcher_id
        self.cursor.execute(delete_sql)
        self.sqlite.commit()

    # 插件卸载时触发
    def __del__(self):
        self.async_loop.close()
        self.cursor.close()
        self.sqlite.close()


# 全局函数


def baiduTranslate(novel_config, translate_text, flag=1) -> str:
    """
    :param translate_text: 待翻译的句子，字数小于2000
    :param flag: 1:英文->中文; 0:中文->英文;
    :param novel_config: novelAI配置文件
    :return: 成功：返回服务器结果。失败：返回服务器失败原因。
    """
    baidu_trans_conf = novel_config.get('Translate').get('baidu')
    api_key = baidu_trans_conf.get('apikey')
    api_secret = baidu_trans_conf.get('api_secret')
    # 获取百度access_token
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": api_key, "client_secret": api_secret}
    access_token = str(requests.post(url, params=params).json().get("access_token"))
    # 开始翻译
    url = "https://aip.baidubce.com/rpc/2.0/mt/texttrans/v1?access_token=" + access_token
    if flag:
        payload = json.dumps({
            "from": "en",
            "to": "zh",
            "q": translate_text
        })
    else:
        payload = json.dumps({
            "from": "zh",
            "to": "en",
            "q": translate_text
        })
    baidu_headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    resp = requests.request("POST", url, headers=baidu_headers, data=payload)
    resp_json = json.loads(resp.content)
    if resp_json.get('error_code'):
        logging.error(
            f"[NovelAi]: 百度翻译错误，错误码：{resp_json.get('error_code')}，错误信息：{resp_json.get('error_msg')}")
    return resp_json.get('result').get('trans_result')[0].get('dst')


def googleTranslate(novel_config, translate_text, flag=1) -> str:
    """谷歌机器翻译,或许需要挂系统代理？
  :param translate_text: 待翻译的句子，字数小于15K
  :param flag: 1:英文->中文; 0:中文->英文;
  :param novel_config: novelAI配置文件
  :return: 成功：返回服务器结果。失败：你猜猜会怎么样？
  """
    from pygoogletranslation import Translator
    # 翻译对象
    translator = Translator()
    # 获取最大尝试次数
    max_number = novel_config.get("Translate").get("google").get("try_number")
    if max_number is None:
        max_number = 10
    # 判断中英文
    if flag:
        language_dest = 'en'
    else:
        language_dest = 'zh-cn'
    # 翻译处理
    t = 0
    if max_number is None:
        max_number = 10
    # 请求翻译，直到成功或超出max_number
    while True:
        try:
            t = (translator.translate([f"{translate_text}", "."], dest=language_dest))
        except Exception as e:
            t += 1
            logging.warning(f"[NovelAi]: 谷歌第{t}次翻译错误：{e}\n[NovelAi]: 正在尝试第{t + 1}次")
        if type(t) is not int or t > max_number:
            break
    return t[0].text


def translate_chinese_check(trans_choice, translate_text, flag, novel_config) -> str:
    """翻译接口的判断，判断使用哪一个翻译接口
    :param translate_text: 要翻译的文本
    :param trans_choice: 选择的翻译接口: "百度" "谷歌"
    :param flag: 1:英文->中文; 0:中文->英文;
    :param novel_config: novelAI配置文件
    :return: 翻译后的文本
    """
    if re.match('^baidu|^百度', trans_choice):
        en_trans_msg = baiduTranslate(novel_config, translate_text, flag)
    elif re.match('^google|^谷歌', trans_choice):
        en_trans_msg = googleTranslate(novel_config, translate_text, flag)
    else:
        logging.warning("[NovelAi]: 无法获取到你选择的翻译,默认为你使用谷歌翻译")
        en_trans_msg = NovalAiStoryPlugins.googleTranslate(novel_config, translate_text, flag)
    return en_trans_msg
