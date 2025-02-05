# Воспроизведение экспериментов

Для настройки среды следуйте инструкциям из `README.md`

Для воспроизведения экспериментов необходимо запустить файл:
```bash
python run_experiment1.py run_name='name of the run'
```

Конфигурации эксперимента настраиваются в файде `configs/exp1.yaml` (можно подробнее ознакомиться с файлом). Ключевые параметры, которые я изменял в экспериментах:
* style_guider_scale_multiplier - множитель на который домножается guider_scale из конфига с гайдерами
* style_guider_scale_default - дефолтное значение guider_scale 
т. е. итоговое значение `guider_scale` для конкретного гайдера = `style_guider_scale_multiplier` * `style_guider_scale_default`. По ключу `samples` для кадждого семпла можно определить промпты для I_cnt, I_sty, и edit prompt (подробнее про эксперименты с edit prompt в pdf отчете).