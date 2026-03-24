#!/usr/bin/env python3
"""Generate on-device assistant tool-calling training data with Gemini and merge
into the existing unified dataset.

Usage:
    GEMINI_API_KEY=... python scripts/generate_data.py --num-samples 5000
    GEMINI_API_KEY=... python scripts/generate_data.py --num-samples 100 --dry-run
    GEMINI_API_KEY=... python scripts/generate_data.py --num-samples 5000 --workers 32
"""

import argparse
import concurrent.futures
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor

from google import genai
from tqdm import tqdm

POOL_TIME_PRODUCTIVITY = [
    {"name": "set_timer", "description": "Set a timer for the specified duration or end time.", "parameters": {"time_human": {"type": "string", "description": "The duration or target end time in human readable format e.g. '1 hour and 30 minutes', '45 minutes', 'for 1:50pm', 'at 13:30'.", "required": True}}},
    {"name": "set_alarm", "description": "Set an alarm for a specified time.", "parameters": {"time_hours": {"type": "number", "description": "The hour component of the alarm time (24 hour time).", "required": True}, "time_minutes": {"type": "number", "description": "The minute component of the alarm time (0-59).", "required": True}, "label": {"type": "string", "description": "An optional label or title for the alarm.", "required": False}}},
    {"name": "create_reminder", "description": "Create a reminder for the user at a specific time.", "parameters": {"message": {"type": "string", "description": "The reminder message e.g. 'Buy more milk'.", "required": True}, "date_time_human": {"type": "string", "description": "The date and/or time for the reminder in human readable format e.g. 'tomorrow at 13:00', 'next Monday at 9am'.", "required": False}}},
    {"name": "create_calendar_event", "description": "Create a new calendar event with a title, start time, and optional end time.", "parameters": {"title": {"type": "string", "description": "The title of the calendar event.", "required": True}, "start_time_human": {"type": "string", "description": "The start date/time in human readable format e.g. 'tomorrow at 2pm', 'March 15 at 10:00'.", "required": True}, "end_time_human": {"type": "string", "description": "The end date/time in human readable format.", "required": False}, "location": {"type": "string", "description": "The location of the event.", "required": False}}},
    {"name": "stop_timer", "description": "Stop the currently running timer.", "parameters": {}},
    {"name": "snooze_alarm", "description": "Snooze the currently ringing alarm for a specified duration.", "parameters": {"minutes": {"type": "number", "description": "Number of minutes to snooze (default 5).", "required": False}}},
    {"name": "delete_alarm", "description": "Delete an existing alarm.", "parameters": {"label": {"type": "string", "description": "Label of the alarm to delete.", "required": False}, "time_hours": {"type": "number", "description": "Hour of the alarm to delete.", "required": False}}},
    {"name": "get_next_alarm", "description": "Get the time of the next scheduled alarm.", "parameters": {}},
    {"name": "delete_reminder", "description": "Delete a reminder by matching its message.", "parameters": {"search_text": {"type": "string", "description": "Text to match the reminder to delete.", "required": True}}},
    {"name": "get_upcoming_events", "description": "Get upcoming calendar events for today or a specified date.", "parameters": {"date_human": {"type": "string", "description": "Date in human readable format e.g. 'today', 'tomorrow', 'next Friday'. Defaults to today.", "required": False}}},
    {"name": "delete_calendar_event", "description": "Delete a calendar event by title.", "parameters": {"title": {"type": "string", "description": "Title of the event to delete.", "required": True}}},
    {"name": "start_stopwatch", "description": "Start a stopwatch.", "parameters": {}},
    {"name": "stop_stopwatch", "description": "Stop the running stopwatch and return elapsed time.", "parameters": {}},
    {"name": "get_current_time", "description": "Get the current date and time.", "parameters": {"timezone": {"type": "string", "description": "Optional timezone e.g. 'America/New_York', 'Europe/London'.", "required": False}}},
    {"name": "start_focus_mode", "description": "Start a focus/concentration session blocking notifications.", "parameters": {"duration_minutes": {"type": "number", "description": "Duration of focus session in minutes.", "required": True}, "allow_calls": {"type": "boolean", "description": "Whether to allow phone calls during focus.", "required": False}}},
]

POOL_LISTS_NOTES = [
    {"name": "create_list_item", "description": "Add a new item to a user's list (shopping, todo, etc.) with an optional reminder.", "parameters": {"list_name": {"type": "string", "description": "Short name of the list e.g. 'shopping', 'todo', 'groceries'.", "required": True}, "message": {"type": "string", "description": "The text of the list item.", "required": True}, "reminder_date_time_human": {"type": "string", "description": "Optional reminder date/time in human readable format.", "required": False}}},
    {"name": "create_note", "description": "Create a new note with the given text.", "parameters": {"text": {"type": "string", "description": "The text of the note.", "required": True}, "title": {"type": "string", "description": "Optional title for the note.", "required": False}}},
    {"name": "delete_list_item", "description": "Remove an item from a list by searching for it.", "parameters": {"list_name": {"type": "string", "description": "The name of the list.", "required": True}, "search_text": {"type": "string", "description": "Text to match the item to delete.", "required": True}}},
    {"name": "get_list_items", "description": "Retrieve all items from a specified list.", "parameters": {"list_name": {"type": "string", "description": "The name of the list to retrieve.", "required": True}}},
    {"name": "create_list", "description": "Create a new empty list.", "parameters": {"list_name": {"type": "string", "description": "Name for the new list.", "required": True}}},
    {"name": "delete_list", "description": "Delete an entire list.", "parameters": {"list_name": {"type": "string", "description": "Name of the list to delete.", "required": True}}},
    {"name": "search_notes", "description": "Search through existing notes by keyword.", "parameters": {"query": {"type": "string", "description": "Search keyword or phrase.", "required": True}}},
    {"name": "delete_note", "description": "Delete a note by matching its title or content.", "parameters": {"search_text": {"type": "string", "description": "Text to match the note to delete.", "required": True}}},
    {"name": "mark_list_item_done", "description": "Mark a list item as completed.", "parameters": {"list_name": {"type": "string", "description": "The list name.", "required": True}, "search_text": {"type": "string", "description": "Text to match the item to mark done.", "required": True}}},
    {"name": "share_list", "description": "Share a list with another contact.", "parameters": {"list_name": {"type": "string", "description": "The list to share.", "required": True}, "contact_id": {"type": "string", "description": "The contact to share with.", "required": True}}},
]

POOL_MESSAGING = [
    {"name": "send_instant_message", "description": "Send an instant message to a contact.", "parameters": {"contact_id": {"type": "string", "description": "The unique identifier of the recipient contact.", "required": True}, "text": {"type": "string", "description": "The message text to send.", "required": True}}},
    {"name": "search_for_contact", "description": "Search for a contact by name to get their contact ID.", "parameters": {"name": {"type": "string", "description": "The name of the contact to search for.", "required": True}}},
    {"name": "make_phone_call", "description": "Initiate a phone call to a contact.", "parameters": {"contact_id": {"type": "string", "description": "The unique identifier of the contact to call.", "required": True}}},
    {"name": "send_email", "description": "Send an email to a recipient.", "parameters": {"to": {"type": "string", "description": "The recipient's email address.", "required": True}, "subject": {"type": "string", "description": "The email subject line.", "required": True}, "body": {"type": "string", "description": "The email body text.", "required": True}}},
    {"name": "send_sms", "description": "Send an SMS text message to a phone number.", "parameters": {"phone_number": {"type": "string", "description": "The recipient phone number.", "required": True}, "text": {"type": "string", "description": "The message text.", "required": True}}},
    {"name": "get_recent_messages", "description": "Get recent messages from a contact or all contacts.", "parameters": {"contact_id": {"type": "string", "description": "Optional contact ID to filter messages.", "required": False}, "count": {"type": "number", "description": "Number of recent messages to retrieve (default 10).", "required": False}}},
    {"name": "get_call_history", "description": "Get recent call history.", "parameters": {"count": {"type": "number", "description": "Number of recent calls to show (default 10).", "required": False}}},
    {"name": "create_contact", "description": "Create a new contact entry.", "parameters": {"name": {"type": "string", "description": "Full name of the contact.", "required": True}, "phone": {"type": "string", "description": "Phone number.", "required": False}, "email": {"type": "string", "description": "Email address.", "required": False}}},
    {"name": "block_contact", "description": "Block or unblock a contact.", "parameters": {"contact_id": {"type": "string", "description": "The contact to block/unblock.", "required": True}, "action": {"type": "string", "description": "'block' or 'unblock'.", "required": True}}},
    {"name": "start_video_call", "description": "Start a video call with a contact.", "parameters": {"contact_id": {"type": "string", "description": "The contact to video call.", "required": True}}},
]

POOL_DEVICE_CONTROL = [
    {"name": "set_brightness", "description": "Set the screen brightness level.", "parameters": {"level": {"type": "number", "description": "Brightness level from 0 to 100.", "required": True}}},
    {"name": "set_volume", "description": "Set the device volume level.", "parameters": {"level": {"type": "number", "description": "Volume level from 0 to 100.", "required": True}, "stream": {"type": "string", "description": "Which audio stream: 'media', 'ringtone', 'alarm', 'notification'.", "required": False}}},
    {"name": "toggle_wifi", "description": "Turn Wi-Fi on or off.", "parameters": {"enabled": {"type": "boolean", "description": "True to enable, false to disable.", "required": True}}},
    {"name": "toggle_bluetooth", "description": "Turn Bluetooth on or off.", "parameters": {"enabled": {"type": "boolean", "description": "True to enable, false to disable.", "required": True}}},
    {"name": "toggle_flashlight", "description": "Turn the flashlight on or off.", "parameters": {"enabled": {"type": "boolean", "description": "True to turn on, false to turn off.", "required": True}}},
    {"name": "toggle_do_not_disturb", "description": "Enable or disable Do Not Disturb mode.", "parameters": {"enabled": {"type": "boolean", "description": "True to enable DND, false to disable.", "required": True}, "duration_minutes": {"type": "number", "description": "Optional duration in minutes before auto-disabling.", "required": False}}},
    {"name": "set_screen_timeout", "description": "Set the screen auto-lock timeout duration.", "parameters": {"seconds": {"type": "number", "description": "Timeout in seconds (e.g. 30, 60, 120, 300).", "required": True}}},
    {"name": "toggle_auto_brightness", "description": "Enable or disable automatic brightness adjustment.", "parameters": {"enabled": {"type": "boolean", "description": "True to enable, false to disable.", "required": True}}},
    {"name": "toggle_auto_rotate", "description": "Enable or disable auto screen rotation.", "parameters": {"enabled": {"type": "boolean", "description": "True to enable, false to disable.", "required": True}}},
    {"name": "toggle_hotspot", "description": "Enable or disable the mobile hotspot.", "parameters": {"enabled": {"type": "boolean", "description": "True to enable, false to disable.", "required": True}, "password": {"type": "string", "description": "Optional hotspot password.", "required": False}}},
    {"name": "toggle_power_saving", "description": "Enable or disable battery power saving mode.", "parameters": {"enabled": {"type": "boolean", "description": "True to enable, false to disable.", "required": True}}},
    {"name": "connect_bluetooth_device", "description": "Connect to a specific Bluetooth device by name.", "parameters": {"device_name": {"type": "string", "description": "Name of the Bluetooth device e.g. 'AirPods Pro', 'JBL Speaker'.", "required": True}}},
    {"name": "connect_wifi_network", "description": "Connect to a specific Wi-Fi network.", "parameters": {"ssid": {"type": "string", "description": "Name of the Wi-Fi network.", "required": True}, "password": {"type": "string", "description": "Wi-Fi password if required.", "required": False}}},
    {"name": "toggle_night_mode", "description": "Enable or disable night/blue light filter.", "parameters": {"enabled": {"type": "boolean", "description": "True to enable, false to disable.", "required": True}}},
]

POOL_MEDIA = [
    {"name": "play_music", "description": "Play music by song name, artist, album, or genre.", "parameters": {"query": {"type": "string", "description": "Search query: song title, artist name, album, or genre.", "required": True}}},
    {"name": "pause_media", "description": "Pause the currently playing media.", "parameters": {}},
    {"name": "resume_media", "description": "Resume the paused media playback.", "parameters": {}},
    {"name": "skip_track", "description": "Skip to the next track in the current playlist.", "parameters": {}},
    {"name": "previous_track", "description": "Go back to the previous track.", "parameters": {}},
    {"name": "play_podcast", "description": "Play a podcast by name or topic.", "parameters": {"query": {"type": "string", "description": "Podcast name or topic to search for.", "required": True}}},
    {"name": "play_radio", "description": "Play a radio station by name or frequency.", "parameters": {"station": {"type": "string", "description": "Station name or frequency e.g. 'NPR', '101.1 FM'.", "required": True}}},
    {"name": "play_audiobook", "description": "Play an audiobook by title or author.", "parameters": {"query": {"type": "string", "description": "Audiobook title or author name.", "required": True}}},
    {"name": "set_repeat_mode", "description": "Set the media repeat mode.", "parameters": {"mode": {"type": "string", "description": "'off', 'one', or 'all'.", "required": True}}},
    {"name": "set_shuffle", "description": "Enable or disable shuffle playback.", "parameters": {"enabled": {"type": "boolean", "description": "True to enable shuffle, false to disable.", "required": True}}},
    {"name": "get_now_playing", "description": "Get info about the currently playing media.", "parameters": {}},
    {"name": "add_to_playlist", "description": "Add the current song to a playlist.", "parameters": {"playlist_name": {"type": "string", "description": "Name of the playlist.", "required": True}}},
    {"name": "play_playlist", "description": "Play a specific playlist by name.", "parameters": {"playlist_name": {"type": "string", "description": "Name of the playlist to play.", "required": True}}},
    {"name": "cast_media", "description": "Cast current media to a speaker or TV.", "parameters": {"device_name": {"type": "string", "description": "Name of the cast target device e.g. 'Living Room TV', 'Kitchen Speaker'.", "required": True}}},
]

POOL_NAVIGATION = [
    {"name": "get_directions", "description": "Get directions to a destination.", "parameters": {"destination": {"type": "string", "description": "The destination address or place name.", "required": True}, "mode": {"type": "string", "description": "Travel mode: 'driving', 'walking', 'transit', 'cycling'.", "required": False}}},
    {"name": "share_location", "description": "Share current location with a contact.", "parameters": {"contact_id": {"type": "string", "description": "The contact to share location with.", "required": True}, "duration_minutes": {"type": "number", "description": "How long to share location in minutes.", "required": False}}},
    {"name": "find_nearby", "description": "Search for nearby places by category.", "parameters": {"category": {"type": "string", "description": "Type of place e.g. 'gas station', 'restaurant', 'pharmacy', 'coffee shop'.", "required": True}}},
    {"name": "get_eta", "description": "Get estimated time of arrival to a destination.", "parameters": {"destination": {"type": "string", "description": "The destination.", "required": True}, "mode": {"type": "string", "description": "Travel mode.", "required": False}}},
    {"name": "save_location", "description": "Save a location as a favorite or bookmark.", "parameters": {"name": {"type": "string", "description": "Name for the saved location e.g. 'Home', 'Work', 'Gym'.", "required": True}, "address": {"type": "string", "description": "The address to save.", "required": True}}},
    {"name": "get_traffic", "description": "Get current traffic conditions for a route.", "parameters": {"destination": {"type": "string", "description": "The destination to check traffic for.", "required": True}}},
    {"name": "start_navigation", "description": "Start turn-by-turn navigation to a destination.", "parameters": {"destination": {"type": "string", "description": "The destination.", "required": True}, "mode": {"type": "string", "description": "Travel mode.", "required": False}, "avoid_tolls": {"type": "boolean", "description": "Whether to avoid toll roads.", "required": False}}},
    {"name": "stop_navigation", "description": "Stop the current navigation.", "parameters": {}},
    {"name": "report_traffic_incident", "description": "Report a traffic incident at current location.", "parameters": {"type": {"type": "string", "description": "Type of incident: 'accident', 'construction', 'hazard', 'police'.", "required": True}}},
]

POOL_SMART_HOME = [
    {"name": "set_thermostat", "description": "Set the home thermostat to a target temperature.", "parameters": {"temperature": {"type": "number", "description": "Target temperature in degrees.", "required": True}, "unit": {"type": "string", "description": "'fahrenheit' or 'celsius'.", "required": False}}},
    {"name": "control_lights", "description": "Turn lights on or off in a specific room, optionally setting brightness or color.", "parameters": {"room": {"type": "string", "description": "The room name e.g. 'bedroom', 'kitchen', 'living room'.", "required": True}, "action": {"type": "string", "description": "'on', 'off', or 'dim'.", "required": True}, "brightness": {"type": "number", "description": "Brightness 0-100 (for 'on' or 'dim').", "required": False}, "color": {"type": "string", "description": "Optional light color e.g. 'warm white', 'cool white', 'red', 'blue'.", "required": False}}},
    {"name": "lock_door", "description": "Lock or unlock a smart door lock.", "parameters": {"door": {"type": "string", "description": "Which door e.g. 'front door', 'back door', 'garage'.", "required": True}, "action": {"type": "string", "description": "'lock' or 'unlock'.", "required": True}}},
    {"name": "start_robot_vacuum", "description": "Start or stop the robot vacuum cleaner.", "parameters": {"action": {"type": "string", "description": "'start', 'stop', or 'dock'.", "required": True}, "room": {"type": "string", "description": "Optional specific room to clean.", "required": False}}},
    {"name": "control_fan", "description": "Control a smart fan.", "parameters": {"room": {"type": "string", "description": "Room where the fan is.", "required": True}, "action": {"type": "string", "description": "'on', 'off'.", "required": True}, "speed": {"type": "string", "description": "'low', 'medium', 'high'.", "required": False}}},
    {"name": "control_blinds", "description": "Open or close smart blinds/curtains.", "parameters": {"room": {"type": "string", "description": "Room name.", "required": True}, "action": {"type": "string", "description": "'open', 'close', or a percentage (0-100).", "required": True}}},
    {"name": "arm_security_system", "description": "Arm or disarm the home security system.", "parameters": {"action": {"type": "string", "description": "'arm_home', 'arm_away', or 'disarm'.", "required": True}}},
    {"name": "get_security_camera_feed", "description": "View a security camera feed.", "parameters": {"camera_name": {"type": "string", "description": "Name of the camera e.g. 'front porch', 'backyard', 'driveway'.", "required": True}}},
    {"name": "control_sprinklers", "description": "Start or stop the garden sprinkler system.", "parameters": {"action": {"type": "string", "description": "'start' or 'stop'.", "required": True}, "zone": {"type": "string", "description": "Optional zone name e.g. 'front yard', 'backyard'.", "required": False}, "duration_minutes": {"type": "number", "description": "How long to run in minutes.", "required": False}}},
    {"name": "set_light_scene", "description": "Activate a predefined lighting scene.", "parameters": {"scene_name": {"type": "string", "description": "Scene name e.g. 'movie', 'dinner', 'bedtime', 'morning', 'party'.", "required": True}}},
    {"name": "control_garage_door", "description": "Open or close the garage door.", "parameters": {"action": {"type": "string", "description": "'open' or 'close'.", "required": True}}},
    {"name": "get_indoor_temperature", "description": "Get the current indoor temperature reading.", "parameters": {"room": {"type": "string", "description": "Optional room name for a specific sensor.", "required": False}}},
]

POOL_UTILITY = [
    {"name": "evaluate_js", "description": "Evaluate a JavaScript expression for calculations, math, date, or string manipulation. Use console.log() to output.", "parameters": {"js": {"type": "string", "description": "The JavaScript code to evaluate.", "required": True}}},
    {"name": "web_search", "description": "Search the web for information.", "parameters": {"query": {"type": "string", "description": "The search query.", "required": True}}},
    {"name": "take_screenshot", "description": "Take a screenshot of the current screen.", "parameters": {}},
    {"name": "open_app", "description": "Open an application by name.", "parameters": {"app_name": {"type": "string", "description": "The name of the app to open.", "required": True}}},
    {"name": "close_app", "description": "Close a running application.", "parameters": {"app_name": {"type": "string", "description": "The name of the app to close.", "required": True}}},
    {"name": "set_wallpaper", "description": "Set the device wallpaper from a URL or built-in option.", "parameters": {"source": {"type": "string", "description": "URL or built-in wallpaper name.", "required": True}}},
    {"name": "translate_text", "description": "Translate text from one language to another.", "parameters": {"text": {"type": "string", "description": "The text to translate.", "required": True}, "target_language": {"type": "string", "description": "Target language code e.g. 'es', 'fr', 'de', 'ja'.", "required": True}, "source_language": {"type": "string", "description": "Source language code (auto-detected if omitted).", "required": False}}},
    {"name": "get_weather", "description": "Get current weather or forecast for a location.", "parameters": {"location": {"type": "string", "description": "City name or location.", "required": True}}},
    {"name": "get_weather_forecast", "description": "Get a multi-day weather forecast.", "parameters": {"location": {"type": "string", "description": "City name or location.", "required": True}, "days": {"type": "number", "description": "Number of forecast days (1-7).", "required": False}}},
    {"name": "set_clipboard", "description": "Copy text to the clipboard.", "parameters": {"text": {"type": "string", "description": "Text to copy.", "required": True}}},
    {"name": "get_clipboard", "description": "Get the current clipboard contents.", "parameters": {}},
    {"name": "define_word", "description": "Look up the definition of a word.", "parameters": {"word": {"type": "string", "description": "The word to define.", "required": True}}},
    {"name": "unit_convert", "description": "Convert between units of measurement.", "parameters": {"value": {"type": "number", "description": "The value to convert.", "required": True}, "from_unit": {"type": "string", "description": "Source unit e.g. 'miles', 'kg', 'fahrenheit'.", "required": True}, "to_unit": {"type": "string", "description": "Target unit e.g. 'km', 'lbs', 'celsius'.", "required": True}}},
    {"name": "scan_qr_code", "description": "Open the camera to scan a QR code.", "parameters": {}},
    {"name": "create_qr_code", "description": "Generate a QR code for given text or URL.", "parameters": {"content": {"type": "string", "description": "Text or URL to encode.", "required": True}}},
]

POOL_CAMERA_PHOTOS = [
    {
        "name": "take_photo",
        "description": "Take a photo using the device camera.",
        "parameters": {
            "camera": {
                "type": "string",
                "description": "'front' or 'back' camera.",
                "required": False,
            },
            "timer_seconds": {
                "type": "number",
                "description": "Optional countdown timer in seconds before taking the photo.",
                "required": False,
            },
        },
    },
    {
        "name": "record_video",
        "description": "Start or stop recording a video.",
        "parameters": {
            "action": {
                "type": "string",
                "description": "'start' or 'stop'.",
                "required": True,
            },
            "camera": {
                "type": "string",
                "description": "'front' or 'back' camera.",
                "required": False,
            },
        },
    },
    {
        "name": "open_gallery",
        "description": "Open the photo gallery or a specific album.",
        "parameters": {
            "album": {
                "type": "string",
                "description": "Optional album name to open.",
                "required": False,
            },
        },
    },
    {
        "name": "share_photo",
        "description": "Share the most recent photo or a specified photo with a contact.",
        "parameters": {
            "contact_id": {
                "type": "string",
                "description": "The contact to share with.",
                "required": True,
            },
            "photo_description": {
                "type": "string",
                "description": "Description to identify which photo, e.g. 'last photo', 'screenshot'.",
                "required": False,
            },
        },
    },
]

POOL_FITNESS_HEALTH = [
    {
        "name": "start_workout",
        "description": "Start tracking a workout activity.",
        "parameters": {
            "workout_type": {
                "type": "string",
                "description": "Type of workout e.g. 'running', 'cycling', 'walking', 'strength', 'yoga'.",
                "required": True,
            },
        },
    },
    {
        "name": "stop_workout",
        "description": "Stop the currently active workout tracking.",
        "parameters": {},
    },
    {
        "name": "log_water_intake",
        "description": "Log water consumption.",
        "parameters": {
            "amount_ml": {
                "type": "number",
                "description": "Amount of water in milliliters.",
                "required": True,
            },
        },
    },
    {
        "name": "get_step_count",
        "description": "Get the current step count for today.",
        "parameters": {},
    },
    {
        "name": "start_sleep_tracking",
        "description": "Start or stop sleep tracking.",
        "parameters": {
            "action": {
                "type": "string",
                "description": "'start' or 'stop'.",
                "required": True,
            },
        },
    },
    {
        "name": "log_meal",
        "description": "Log a meal or food item for nutrition tracking.",
        "parameters": {
            "description": {
                "type": "string",
                "description": "Description of the food or meal.",
                "required": True,
            },
            "meal_type": {
                "type": "string",
                "description": "'breakfast', 'lunch', 'dinner', or 'snack'.",
                "required": False,
            },
        },
    },
]

POOL_SYSTEM = [
    {
        "name": "toggle_airplane_mode",
        "description": "Turn airplane mode on or off.",
        "parameters": {
            "enabled": {
                "type": "boolean",
                "description": "True to enable, false to disable.",
                "required": True,
            },
        },
    },
    {
        "name": "clear_notifications",
        "description": "Clear all or specific app notifications.",
        "parameters": {
            "app_name": {
                "type": "string",
                "description": "Optional app name to clear notifications for. Clears all if omitted.",
                "required": False,
            },
        },
    },
    {
        "name": "open_settings",
        "description": "Open device settings or a specific settings page.",
        "parameters": {
            "page": {
                "type": "string",
                "description": "Optional settings page e.g. 'wifi', 'bluetooth', 'display', 'battery', 'storage'.",
                "required": False,
            },
        },
    },
    {
        "name": "check_battery",
        "description": "Check the current battery level and charging status.",
        "parameters": {},
    },
    {
        "name": "toggle_dark_mode",
        "description": "Enable or disable dark mode.",
        "parameters": {
            "enabled": {
                "type": "boolean",
                "description": "True for dark mode, false for light mode.",
                "required": True,
            },
        },
    },
    {
        "name": "toggle_location_services",
        "description": "Turn location services on or off.",
        "parameters": {
            "enabled": {
                "type": "boolean",
                "description": "True to enable, false to disable.",
                "required": True,
            },
        },
    },
    {
        "name": "restart_device",
        "description": "Restart or shut down the device.",
        "parameters": {
            "action": {
                "type": "string",
                "description": "'restart' or 'shutdown'.",
                "required": True,
            },
        },
    },
]

POOL_FINANCE = [
    {
        "name": "send_payment",
        "description": "Send a payment to a contact.",
        "parameters": {
            "contact_id": {
                "type": "string",
                "description": "The recipient contact ID.",
                "required": True,
            },
            "amount": {
                "type": "number",
                "description": "The amount to send.",
                "required": True,
            },
            "currency": {
                "type": "string",
                "description": "Currency code e.g. 'USD', 'EUR', 'GBP'.",
                "required": False,
            },
            "note": {
                "type": "string",
                "description": "Optional payment note e.g. 'for dinner'.",
                "required": False,
            },
        },
    },
    {
        "name": "check_balance",
        "description": "Check the current account balance.",
        "parameters": {
            "account": {
                "type": "string",
                "description": "Account name e.g. 'checking', 'savings', 'credit card'.",
                "required": False,
            },
        },
    },
    {
        "name": "convert_currency",
        "description": "Convert an amount between currencies.",
        "parameters": {
            "amount": {
                "type": "number",
                "description": "The amount to convert.",
                "required": True,
            },
            "from_currency": {
                "type": "string",
                "description": "Source currency code e.g. 'USD'.",
                "required": True,
            },
            "to_currency": {
                "type": "string",
                "description": "Target currency code e.g. 'EUR'.",
                "required": True,
            },
        },
    },
]

POOL_READING_NEWS = [
    {
        "name": "get_news_headlines",
        "description": "Get the latest news headlines, optionally filtered by topic.",
        "parameters": {
            "topic": {
                "type": "string",
                "description": "Optional topic filter e.g. 'technology', 'sports', 'politics', 'business'.",
                "required": False,
            },
        },
    },
    {
        "name": "read_aloud",
        "description": "Read text content aloud using text-to-speech.",
        "parameters": {
            "text": {
                "type": "string",
                "description": "The text to read aloud.",
                "required": True,
            },
            "speed": {
                "type": "number",
                "description": "Speech rate multiplier (0.5 = slow, 1.0 = normal, 2.0 = fast).",
                "required": False,
            },
        },
    },
    {
        "name": "summarize_page",
        "description": "Summarize the content of the currently open web page or article.",
        "parameters": {},
    },
]

POOL_ACCESSIBILITY = [
    {
        "name": "set_font_size",
        "description": "Set the system font size.",
        "parameters": {
            "size": {
                "type": "string",
                "description": "'small', 'medium', 'large', or 'extra_large'.",
                "required": True,
            },
        },
    },
    {
        "name": "toggle_voice_assistant",
        "description": "Enable or disable the voice assistant listener.",
        "parameters": {
            "enabled": {
                "type": "boolean",
                "description": "True to enable, false to disable.",
                "required": True,
            },
        },
    },
    {
        "name": "toggle_magnifier",
        "description": "Enable or disable the screen magnifier.",
        "parameters": {
            "enabled": {
                "type": "boolean",
                "description": "True to enable, false to disable.",
                "required": True,
            },
        },
    },
    {
        "name": "toggle_screen_reader",
        "description": "Enable or disable the screen reader for accessibility.",
        "parameters": {
            "enabled": {
                "type": "boolean",
                "description": "True to enable, false to disable.",
                "required": True,
            },
        },
    },
]

POOL_SHOPPING = [
    {
        "name": "search_product",
        "description": "Search for a product to buy online.",
        "parameters": {
            "query": {
                "type": "string",
                "description": "Product search query.",
                "required": True,
            },
        },
    },
    {
        "name": "add_to_cart",
        "description": "Add a product to the shopping cart.",
        "parameters": {
            "product_name": {
                "type": "string",
                "description": "Name or description of the product.",
                "required": True,
            },
            "quantity": {
                "type": "number",
                "description": "Number of items to add.",
                "required": False,
            },
        },
    },
    {
        "name": "check_order_status",
        "description": "Check the status of a recent order.",
        "parameters": {
            "order_id": {
                "type": "string",
                "description": "The order ID to check. If omitted, checks the most recent order.",
                "required": False,
            },
        },
    },
]

POOL_SOCIAL = [
    {
        "name": "post_status_update",
        "description": "Post a status update or message to social media.",
        "parameters": {
            "text": {
                "type": "string",
                "description": "The status text to post.",
                "required": True,
            },
            "platform": {
                "type": "string",
                "description": "Social platform e.g. 'twitter', 'facebook', 'instagram'.",
                "required": False,
            },
        },
    },
    {
        "name": "check_social_notifications",
        "description": "Check recent social media notifications.",
        "parameters": {
            "platform": {
                "type": "string",
                "description": "Optional platform filter.",
                "required": False,
            },
        },
    },
]

ALL_POOLS = [
    POOL_TIME_PRODUCTIVITY,
    POOL_LISTS_NOTES,
    POOL_MESSAGING,
    POOL_DEVICE_CONTROL,
    POOL_MEDIA,
    POOL_NAVIGATION,
    POOL_SMART_HOME,
    POOL_UTILITY,
    POOL_CAMERA_PHOTOS,
    POOL_FITNESS_HEALTH,
    POOL_SYSTEM,
    POOL_FINANCE,
    POOL_READING_NEWS,
    POOL_ACCESSIBILITY,
    POOL_SHOPPING,
    POOL_SOCIAL,
]


SCENARIOS = [
    # Time & productivity
    "setting a cooking timer", "morning alarm routine", "pomodoro work timer",
    "reminder to take medication", "reminder to pick up kids from school",
    "scheduling a dentist appointment", "meeting reminder for tomorrow",
    "snoozing an alarm", "stopping a running timer",
    # Lists & notes
    "adding groceries to shopping list", "creating a packing list for vacation",
    "jotting down a quick idea", "adding tasks to a todo list for the week",
    "checking what's on the shopping list", "removing an item from a list",
    "noting down a recipe", "noting a license plate number",
    # Messaging & calls
    "texting a friend about dinner plans", "calling mom",
    "sending a work email about a deadline", "looking up a contact then messaging them",
    "sending a birthday greeting", "replying to a message",
    # Device control
    "turning down brightness at night", "turning off wifi to save battery",
    "enabling do not disturb for a meeting", "turning on the flashlight",
    "turning up the volume", "connecting bluetooth headphones",
    "setting screen timeout", "turning on airplane mode",
    # Media
    "playing a specific song", "pausing music", "skipping a track",
    "playing a jazz playlist", "listening to a true crime podcast",
    "resuming a podcast", "playing white noise for sleep",
    # Navigation
    "getting directions to the airport", "finding nearby coffee shops",
    "sharing location with a friend", "finding the nearest gas station",
    "walking directions to the park", "transit directions to downtown",
    # Smart home
    "turning off bedroom lights", "setting thermostat to 72 degrees",
    "locking the front door before bed", "dimming living room lights",
    "starting the robot vacuum", "turning on kitchen lights",
    # Utility
    "calculating a restaurant tip", "converting miles to kilometers",
    "translating a phrase to Spanish", "checking the weather",
    "opening the camera app", "searching the web for a recipe",
    "calculating days until a date", "taking a screenshot",
    # Multi-tool combos
    "setting an alarm AND a reminder", "messaging someone AND setting a timer",
    "turning off lights AND locking doors before bed",
    "adding to shopping list AND setting a reminder to go shopping",
    "looking up a contact AND calling them",
    "getting directions AND sharing location",
    # Camera & photos
    "taking a selfie", "recording a video", "opening the photo gallery",
    "sharing a photo with a friend", "taking a timed photo",
    # Fitness & health
    "starting a run workout", "logging water intake",
    "checking step count for the day", "starting sleep tracking",
    "logging lunch for nutrition", "stopping a workout",
    # System
    "turning on airplane mode for a flight", "clearing all notifications",
    "checking battery level", "turning on dark mode",
    "restarting the phone", "turning off location services",
    "opening wifi settings",
    # Finance
    "sending money to a friend for dinner", "checking bank balance",
    "converting dollars to euros", "paying someone back",
    # Reading & news
    "getting top news headlines", "reading an article aloud",
    "summarizing the current page", "getting sports news",
    # Accessibility
    "making the font bigger", "turning on the screen reader",
    "enabling the magnifier", "turning off voice assistant",
    # Shopping
    "searching for a product online", "adding something to cart",
    "checking order delivery status", "buying a phone case",
    # Social
    "posting a tweet", "checking instagram notifications",
    "posting a status update on facebook",
    # Edge cases
    "very short terse command like 'timer 5 min'",
    "long polite request with extra context",
    "ambiguous request that maps to one tool",
    "casual voice assistant style like 'hey set my alarm for seven'",
    "request where no available tool can help",
]

CALL_TYPES = [
    ("single", "exactly 1 tool call"),
    ("single", "exactly 1 tool call"),
    ("single", "exactly 1 tool call"),
    ("multi", "2-3 tool calls (the user wants multiple things done at once)"),
    ("multi", "2-3 tool calls (the user wants multiple things done at once)"),
    ("none", "NO tool calls — the user asks something none of the available tools can handle, answers must be []"),
    ("no_tools", "NO tool calls — there are NO tools available at all, answers must be []"),
]

MODEL = "gemini-3.1-flash-lite-preview"

MAX_TOOLS = 10


def _pick_tools(rng, force_empty=False):
    """Pick 0-10 tools from 1-3 random pools."""
    if force_empty:
        return []
    num_pools = rng.choice([1, 1, 2, 2, 3])
    pools = rng.sample(ALL_POOLS, num_pools)
    tools = []
    for pool in pools:
        k = rng.randint(max(1, len(pool) // 2), len(pool))
        tools.extend(rng.sample(pool, k))
    rng.shuffle(tools)
    return tools[:MAX_TOOLS]


def build_prompt(batch_size, call_desc, tools, rng):
    """Build a prompt asking Gemini to generate a batch of examples."""
    scenarios_sample = rng.sample(SCENARIOS, min(10, len(SCENARIOS)))
    scenarios_str = "\n".join(f"  - {s}" for s in scenarios_sample)

    if tools:
        tools_section = f"AVAILABLE TOOLS:\n{json.dumps(tools, indent=2)}"
        tool_rules = """- Tool call format: {{"name": "tool_name", "arguments": {{"param": "value"}}}}
- Arguments must match the parameter schemas exactly — use correct types (string, number, boolean)
- Do NOT invent tools not in the list — only use the tools shown above
- For contact_id params, use realistic placeholders like "contact_alice_123", "contact_john_456"
- For evaluate_js, write real working JavaScript with console.log()
- Number params should be actual numbers not strings (e.g. 7 not "7")
- Boolean params should be actual booleans not strings (e.g. true not "true")"""
    else:
        tools_section = "AVAILABLE TOOLS: NONE — no tools are available."
        tool_rules = "- Since no tools are available, ALL answers must be empty arrays []"

    return f"""Generate {batch_size} diverse on-device assistant tool-calling training examples for phones and computers.

{tools_section}

REQUIREMENTS:
- Each example: a "query" (user's natural language) and "answers" (JSON tool calls array)
- This batch: {call_desc}
- Queries should sound like real users talking to a phone/computer voice assistant or typing a quick command
- Vary style: casual ("timer 5 min"), terse ("alarm 7am"), polite ("Could you please..."), conversational ("hey can you set a timer for like 20 minutes")
{tool_rules}
- For empty call examples, answers must be []
- Every query must be UNIQUE — do not repeat patterns

SCENARIO INSPIRATION (vary well beyond these):
{scenarios_str}

OUTPUT FORMAT — return a JSON array, nothing else:
[
  {{
    "query": "user's request",
    "answers": [{{"name": "tool_name", "arguments": {{"param": "value"}}}}]
  }}
]

Return ONLY valid JSON. No markdown, no explanation."""

def make_clients():
    """Create Gemini clients from GEMINI_API_KEY (comma-separated for multiple keys)."""
    raw = os.environ.get("GEMINI_API_KEY", "")
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    if not keys:
        print("Error: GEMINI_API_KEY environment variable not set.", file=sys.stderr)
        print("Set one or more comma-separated keys:", file=sys.stderr)
        print("  export GEMINI_API_KEY=key1,key2,key3", file=sys.stderr)
        print("Get keys at https://aistudio.google.com/apikey", file=sys.stderr)
        sys.exit(1)
    clients = [genai.Client(api_key=k) for k in keys]
    print(f"Using {len(clients)} API key(s)")
    return clients


class ClientPool:
    """Round-robin pool of Gemini clients for distributing requests across API keys."""

    def __init__(self, clients):
        self._clients = clients
        self._idx = 0
        self._lock = __import__("threading").Lock()

    def get(self):
        with self._lock:
            client = self._clients[self._idx % len(self._clients)]
            self._idx += 1
            return client


def generate_batch(client_pool, batch_size, rng, model):
    """Generate one batch of examples. Returns list of dicts."""
    call_type, call_desc = rng.choice(CALL_TYPES)
    tools = _pick_tools(rng, force_empty=(call_type == "no_tools"))

    prompt = build_prompt(batch_size, call_desc, tools, rng)

    client = client_pool.get()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config={
            "temperature": 1.0,
            "max_output_tokens": 16384,
        },
    )

    text = response.text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        examples = json.loads(text)
    except json.JSONDecodeError:
        return []

    if not isinstance(examples, list):
        return []

    valid = []
    tools_str = json.dumps(tools, separators=(",", ":"))
    tool_name_set = {t["name"] for t in tools}

    for ex in examples:
        if not isinstance(ex, dict):
            continue
        query = ex.get("query", "").strip()
        answers = ex.get("answers")
        if not query or answers is None:
            continue
        if not isinstance(answers, list):
            continue

        ok = True
        for call in answers:
            if not isinstance(call, dict):
                ok = False
                break
            if call.get("name") not in tool_name_set:
                ok = False
                break
            if not isinstance(call.get("arguments", {}), dict):
                ok = False
                break
        if not ok:
            continue

        valid.append({
            "query": query,
            "tools": tools_str,
            "answers": json.dumps(answers, separators=(",", ":")),
            "source": "synth-gemini-assistant",
            "model": model,
        })

    return valid


def generate_all(num_samples, workers=8, batch_size=25, model=MODEL, client_pool=None):
    """Generate num_samples examples using parallel Gemini calls."""
    if client_pool is None:
        client_pool = ClientPool(make_clients())
    rng = random.Random(42)

    target = int(num_samples * 1.3)
    num_batches = (target + batch_size - 1) // batch_size
    all_examples = []
    failed = 0

    pbar = tqdm(total=num_batches, desc="Generating", unit="batch")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = set()
        submitted = 0

        def _submit_one():
            nonlocal submitted
            batch_rng = random.Random(rng.randint(0, 2**32))
            f = pool.submit(generate_batch, client_pool, batch_size, batch_rng, model)
            pending.add(f)
            submitted += 1

        for _ in range(min(workers, num_batches)):
            _submit_one()

        while pending:
            done, pending = concurrent.futures.wait(
                pending, return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for f in done:
                try:
                    results = f.result()
                    all_examples.extend(results)
                except Exception as e:
                    failed += 1
                pbar.update(1)
                pbar.set_postfix(examples=len(all_examples), failed=failed)
                if submitted < num_batches:
                    _submit_one()

    pbar.close()

    seen = set()
    deduped = []
    for ex in all_examples:
        if ex["query"] not in seen:
            seen.add(ex["query"])
            deduped.append(ex)

    if len(deduped) > num_samples:
        deduped = deduped[:num_samples]

    print(f"Generated {len(deduped):,} unique examples ({len(all_examples) - len(deduped)} duplicates removed, {failed} failed batches)")
    return deduped



LOCAL_UNIFIED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "tool_calls_unified")
HF_DATASET_REPO = "Cactus-Compute/tool-calls"
UPLOAD_EVERY = 1000


def _load_existing():
    """Load existing dataset from disk or HuggingFace."""
    from datasets import load_dataset, load_from_disk

    local = os.path.abspath(LOCAL_UNIFIED_DIR)
    if os.path.exists(local) and any(f.endswith(".arrow") for f in os.listdir(local)):
        ds = load_from_disk(local)
        print(f"Loaded existing dataset: {len(ds)} rows")
    else:
        print(f"Downloading existing dataset from {HF_DATASET_REPO}...")
        ds = load_dataset(HF_DATASET_REPO, split="train", token=True)
        print(f"Downloaded: {len(ds)} rows")
    return ds


def _merge_and_upload(existing, new_examples):
    """Merge new examples into existing dataset, save locally, and upload."""
    import shutil
    from collections import Counter
    from datasets import Dataset, concatenate_datasets

    new_ds = Dataset.from_dict({
        "query": [ex["query"] for ex in new_examples],
        "tools": [ex["tools"] for ex in new_examples],
        "answers": [ex["answers"] for ex in new_examples],
        "source": [ex["source"] for ex in new_examples],
        "model": [ex["model"] for ex in new_examples],
    })
    new_ds = new_ds.select_columns(existing.column_names)

    merged = concatenate_datasets([existing, new_ds])
    print(f"\nMerged: {len(merged)} rows (+{len(new_ds)} new)")

    for src, cnt in Counter(merged["source"]).most_common():
        print(f"  {src}: {cnt}")

    local = os.path.abspath(LOCAL_UNIFIED_DIR)
    tmp_dir = local + "_tmp"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    merged.save_to_disk(tmp_dir)
    if os.path.exists(local):
        shutil.rmtree(local)
    os.rename(tmp_dir, local)
    print("Saved locally.")

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(HF_DATASET_REPO, repo_type="dataset", private=False, exist_ok=True)
    print(f"Uploading to {HF_DATASET_REPO}...")
    merged.push_to_hub(HF_DATASET_REPO, token=True)
    print(f"Upload complete: {HF_DATASET_REPO}")

    return merged


def main():
    parser = argparse.ArgumentParser(description="Generate on-device assistant tool-calling data with Gemini")
    parser.add_argument("--num-samples", type=int, default=5000, help="Number of samples to generate")
    parser.add_argument("--batch-size", type=int, default=25, help="Examples per Gemini call")
    parser.add_argument("--workers", type=int, default=8, help="Parallel Gemini calls")
    parser.add_argument("--model", type=str, default=MODEL, help="Gemini model")
    parser.add_argument("--dry-run", action="store_true", help="Generate only, skip save and upload")
    parser.add_argument("--output-jsonl", type=str, default=None, help="Also save raw generations to JSONL")
    parser.add_argument("--upload-every", type=int, default=UPLOAD_EVERY, help="Merge+upload every N samples (default 1000)")
    args = parser.parse_args()

    client_pool = ClientPool(make_clients())

    remaining = args.num_samples
    total_generated = 0
    existing = None if args.dry_run else _load_existing()
    seen_queries = set()
    if existing and not args.dry_run:
        seen_queries = set(existing["query"][:50000])

    while remaining > 0:
        chunk_size = min(remaining, args.upload_every)
        print(f"\n{'='*50}")
        print(f"Generating chunk: {chunk_size} samples ({total_generated}/{args.num_samples} done)")
        print(f"{'='*50}")

        examples = generate_all(chunk_size, args.workers, args.batch_size, args.model, client_pool=client_pool)

        if not examples:
            print("No examples generated in this chunk, stopping.")
            break

        fresh = []
        for ex in examples:
            if ex["query"] not in seen_queries:
                seen_queries.add(ex["query"])
                fresh.append(ex)
        if len(fresh) < len(examples):
            print(f"  Removed {len(examples) - len(fresh)} cross-chunk duplicates")
        examples = fresh

        if not examples:
            print("All examples were duplicates, stopping.")
            break

        total_generated += len(examples)
        remaining -= len(examples)

        if args.output_jsonl:
            with open(args.output_jsonl, "a") as f:
                for ex in examples:
                    f.write(json.dumps(ex) + "\n")

        if total_generated == len(examples):
            print("\nSample examples:")
            for ex in examples[:5]:
                print(f"  Q: {ex['query']}")
                print(f"  A: {ex['answers'][:150]}")
                print()

        if args.dry_run:
            print(f"  Chunk: {len(examples)} examples (dry-run, not uploading)")
            continue

        existing = _merge_and_upload(existing, examples)

    print(f"\nDone. Total generated: {total_generated:,}")
    if args.output_jsonl:
        print(f"Raw JSONL: {args.output_jsonl}")


if __name__ == "__main__":
    main()
