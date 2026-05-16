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
import re
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
    {"name": "edit_note", "description": "Edit an existing note by appending or replacing its content.", "parameters": {"search_text": {"type": "string", "description": "Text to match the note to edit.", "required": True}, "new_text": {"type": "string", "description": "New text to append or replace with.", "required": True}, "mode": {"type": "string", "description": "'append' to add to existing or 'replace' to overwrite.", "required": False}}},
    {"name": "pin_note", "description": "Pin or unpin a note so it stays at the top.", "parameters": {"search_text": {"type": "string", "description": "Text to match the note.", "required": True}, "pinned": {"type": "boolean", "description": "True to pin, false to unpin.", "required": True}}},
    {"name": "tag_note", "description": "Add or remove a tag/label on a note.", "parameters": {"search_text": {"type": "string", "description": "Text to match the note.", "required": True}, "tag": {"type": "string", "description": "Tag name e.g. 'work', 'personal', 'urgent', 'ideas'.", "required": True}, "action": {"type": "string", "description": "'add' or 'remove'.", "required": False}}},
    {"name": "archive_note", "description": "Archive or unarchive a note.", "parameters": {"search_text": {"type": "string", "description": "Text to match the note.", "required": True}, "archive": {"type": "boolean", "description": "True to archive, false to unarchive.", "required": True}}},
    {"name": "create_voice_note", "description": "Record a voice note with optional transcription.", "parameters": {"title": {"type": "string", "description": "Optional title for the voice note.", "required": False}, "transcribe": {"type": "boolean", "description": "Whether to automatically transcribe the recording.", "required": False}}},
    {"name": "create_checklist", "description": "Create a note with a checklist of items.", "parameters": {"title": {"type": "string", "description": "Title for the checklist.", "required": True}, "items": {"type": "string", "description": "Comma-separated list of checklist items.", "required": True}}},
    {"name": "rename_list", "description": "Rename an existing list.", "parameters": {"list_name": {"type": "string", "description": "Current name of the list.", "required": True}, "new_name": {"type": "string", "description": "New name for the list.", "required": True}}},
    {"name": "sort_list", "description": "Sort list items by a given criteria.", "parameters": {"list_name": {"type": "string", "description": "The list to sort.", "required": True}, "sort_by": {"type": "string", "description": "'alphabetical', 'date_added', 'priority', or 'completed'.", "required": False}}},
    {"name": "duplicate_list", "description": "Create a copy of an existing list.", "parameters": {"list_name": {"type": "string", "description": "The list to duplicate.", "required": True}, "new_name": {"type": "string", "description": "Name for the duplicate.", "required": False}}},
    {"name": "export_note", "description": "Export a note as a text file or PDF.", "parameters": {"search_text": {"type": "string", "description": "Text to match the note to export.", "required": True}, "format": {"type": "string", "description": "'text' or 'pdf'.", "required": False}}},
    {"name": "get_all_notes", "description": "List all notes, optionally filtered by tag.", "parameters": {"tag": {"type": "string", "description": "Optional tag to filter by.", "required": False}}},
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
    {"name": "take_photo", "description": "Take a photo using the device camera.", "parameters": {"camera": {"type": "string", "description": "'front' or 'back' camera.", "required": False}, "timer_seconds": {"type": "number", "description": "Optional countdown timer in seconds before taking the photo.", "required": False}}},
    {"name": "record_video", "description": "Start or stop recording a video.", "parameters": {"action": {"type": "string", "description": "'start' or 'stop'.", "required": True}, "camera": {"type": "string", "description": "'front' or 'back' camera.", "required": False}}},
    {"name": "open_gallery", "description": "Open the photo gallery or a specific album.", "parameters": {"album": {"type": "string", "description": "Optional album name to open.", "required": False}}},
    {"name": "share_photo", "description": "Share the most recent photo or a specified photo with a contact.", "parameters": {"contact_id": {"type": "string", "description": "The contact to share with.", "required": True}, "photo_description": {"type": "string", "description": "Description to identify which photo, e.g. 'last photo', 'screenshot'.", "required": False}}},
    {"name": "take_panorama", "description": "Take a panoramic photo.", "parameters": {"direction": {"type": "string", "description": "Optional sweep direction e.g. 'left', 'right'.", "required": False}}},
    {"name": "take_burst_photos", "description": "Take a burst of photos in quick succession.", "parameters": {"count": {"type": "number", "description": "Optional number of photos to take in the burst.", "required": False}}},
    {"name": "edit_photo", "description": "Edit a photo with a specified action.", "parameters": {"action": {"type": "string", "description": "'crop', 'rotate', or 'filter'.", "required": True}, "filter_name": {"type": "string", "description": "Optional filter name when action is 'filter' e.g. 'sepia', 'noir', 'vivid'.", "required": False}}},
    {"name": "delete_photo", "description": "Delete a photo by description.", "parameters": {"photo_description": {"type": "string", "description": "Description to identify the photo to delete.", "required": True}}},
    {"name": "create_photo_album", "description": "Create a new photo album.", "parameters": {"album_name": {"type": "string", "description": "Name for the new album.", "required": True}}},
    {"name": "search_photos", "description": "Search photos by keyword or description.", "parameters": {"query": {"type": "string", "description": "Search query e.g. 'beach', 'sunset', 'birthday'.", "required": True}}},
]

POOL_FITNESS_HEALTH = [
    {"name": "start_workout", "description": "Start tracking a workout activity.", "parameters": {"workout_type": {"type": "string", "description": "Type of workout e.g. 'running', 'cycling', 'walking', 'strength', 'yoga'.", "required": True}}},
    {"name": "stop_workout", "description": "Stop the currently active workout tracking.", "parameters": {}},
    {"name": "log_water_intake", "description": "Log water consumption.", "parameters": {"amount_ml": {"type": "number", "description": "Amount of water in milliliters.", "required": True}}},
    {"name": "get_step_count", "description": "Get the current step count for today.", "parameters": {}},
    {"name": "start_sleep_tracking", "description": "Start or stop sleep tracking.", "parameters": {"action": {"type": "string", "description": "'start' or 'stop'.", "required": True}}},
    {"name": "log_meal", "description": "Log a meal or food item for nutrition tracking.", "parameters": {"description": {"type": "string", "description": "Description of the food or meal.", "required": True}, "meal_type": {"type": "string", "description": "'breakfast', 'lunch', 'dinner', or 'snack'.", "required": False}}},
    {"name": "get_heart_rate_history", "description": "Get heart rate history over a period.", "parameters": {"period": {"type": "string", "description": "Optional period e.g. 'today', 'this_week', 'this_month'.", "required": False}}},
    {"name": "set_step_goal", "description": "Set a daily step goal.", "parameters": {"steps": {"type": "number", "description": "The target number of steps per day.", "required": True}}},
    {"name": "get_sleep_summary", "description": "Get a summary of sleep data for a given date.", "parameters": {"date": {"type": "string", "description": "Optional date in human readable format e.g. 'last night', 'yesterday'. Defaults to last night.", "required": False}}},
    {"name": "log_blood_pressure", "description": "Log a blood pressure reading.", "parameters": {"systolic": {"type": "number", "description": "The systolic pressure value.", "required": True}, "diastolic": {"type": "number", "description": "The diastolic pressure value.", "required": True}}},
]

POOL_SYSTEM = [
    {"name": "toggle_airplane_mode", "description": "Turn airplane mode on or off.", "parameters": {"enabled": {"type": "boolean", "description": "True to enable, false to disable.", "required": True}}},
    {"name": "clear_notifications", "description": "Clear all or specific app notifications.", "parameters": {"app_name": {"type": "string", "description": "Optional app name to clear notifications for. Clears all if omitted.", "required": False}}},
    {"name": "open_settings", "description": "Open device settings or a specific settings page.", "parameters": {"page": {"type": "string", "description": "Optional settings page e.g. 'wifi', 'bluetooth', 'display', 'battery', 'storage'.", "required": False}}},
    {"name": "check_battery", "description": "Check the current battery level and charging status.", "parameters": {}},
    {"name": "toggle_dark_mode", "description": "Enable or disable dark mode.", "parameters": {"enabled": {"type": "boolean", "description": "True for dark mode, false for light mode.", "required": True}}},
    {"name": "toggle_location_services", "description": "Turn location services on or off.", "parameters": {"enabled": {"type": "boolean", "description": "True to enable, false to disable.", "required": True}}},
    {"name": "restart_device", "description": "Restart or shut down the device.", "parameters": {"action": {"type": "string", "description": "'restart' or 'shutdown'.", "required": True}}},
]

POOL_FINANCE = [
    {"name": "send_payment", "description": "Send a payment to a contact.", "parameters": {"contact_id": {"type": "string", "description": "The recipient contact ID.", "required": True}, "amount": {"type": "number", "description": "The amount to send.", "required": True}, "currency": {"type": "string", "description": "Currency code e.g. 'USD', 'EUR', 'GBP'.", "required": False}, "note": {"type": "string", "description": "Optional payment note e.g. 'for dinner'.", "required": False}}},
    {"name": "check_balance", "description": "Check the current account balance.", "parameters": {"account": {"type": "string", "description": "Account name e.g. 'checking', 'savings', 'credit card'.", "required": False}}},
    {"name": "convert_currency", "description": "Convert an amount between currencies.", "parameters": {"amount": {"type": "number", "description": "The amount to convert.", "required": True}, "from_currency": {"type": "string", "description": "Source currency code e.g. 'USD'.", "required": True}, "to_currency": {"type": "string", "description": "Target currency code e.g. 'EUR'.", "required": True}}},
    {"name": "request_payment", "description": "Request a payment from a contact.", "parameters": {"contact_id": {"type": "string", "description": "The contact to request payment from.", "required": True}, "amount": {"type": "number", "description": "The amount to request.", "required": True}, "note": {"type": "string", "description": "Optional note for the request.", "required": False}}},
    {"name": "get_recent_transactions", "description": "Get a list of recent transactions.", "parameters": {"count": {"type": "number", "description": "Number of transactions to retrieve.", "required": False}, "account": {"type": "string", "description": "Optional account name to filter by.", "required": False}}},
    {"name": "set_budget", "description": "Set a spending budget for a category.", "parameters": {"category": {"type": "string", "description": "Budget category e.g. 'food', 'entertainment', 'transport'.", "required": True}, "amount": {"type": "number", "description": "The budget amount.", "required": True}, "period": {"type": "string", "description": "Optional budget period e.g. 'weekly', 'monthly'.", "required": False}}},
    {"name": "get_spending_summary", "description": "Get a summary of spending for a period.", "parameters": {"period": {"type": "string", "description": "Optional period e.g. 'today', 'this_week', 'this_month'.", "required": False}}},
    {"name": "pay_bill", "description": "Pay a bill to a biller.", "parameters": {"biller_name": {"type": "string", "description": "Name of the biller e.g. 'electric company', 'internet provider'.", "required": True}, "amount": {"type": "number", "description": "Optional amount to pay. Uses default if omitted.", "required": False}}},
    {"name": "split_bill", "description": "Split a bill among contacts.", "parameters": {"contact_ids": {"type": "string", "description": "Comma-separated contact IDs to split with.", "required": True}, "total_amount": {"type": "number", "description": "The total bill amount to split.", "required": True}, "note": {"type": "string", "description": "Optional note for the split.", "required": False}}},
    {"name": "get_stock_price", "description": "Get the current price of a stock.", "parameters": {"symbol": {"type": "string", "description": "The stock ticker symbol e.g. 'AAPL', 'GOOGL'.", "required": True}}},
]

POOL_READING_NEWS = [
    {"name": "get_news_headlines", "description": "Get the latest news headlines, optionally filtered by topic.", "parameters": {"topic": {"type": "string", "description": "Optional topic filter e.g. 'technology', 'sports', 'politics', 'business'.", "required": False}}},
    {"name": "read_aloud", "description": "Read text content aloud using text-to-speech.", "parameters": {"text": {"type": "string", "description": "The text to read aloud.", "required": True}, "speed": {"type": "number", "description": "Speech rate multiplier (0.5 = slow, 1.0 = normal, 2.0 = fast).", "required": False}}},
    {"name": "summarize_page", "description": "Summarize the content of the currently open web page or article.", "parameters": {}},
    {"name": "save_article", "description": "Save the current article for later reading.", "parameters": {"title": {"type": "string", "description": "Optional title to save the article under.", "required": False}}},
    {"name": "get_saved_articles", "description": "Retrieve the list of saved articles.", "parameters": {}},
    {"name": "subscribe_newsletter", "description": "Subscribe to a newsletter by topic.", "parameters": {"topic": {"type": "string", "description": "The newsletter topic to subscribe to.", "required": True}}},
    {"name": "open_ebook", "description": "Open an ebook by title.", "parameters": {"title": {"type": "string", "description": "The title of the ebook to open.", "required": True}}},
    {"name": "set_reading_goal", "description": "Set a daily reading goal in pages.", "parameters": {"pages_per_day": {"type": "number", "description": "Number of pages to read per day.", "required": True}}},
    {"name": "get_reading_progress", "description": "Get current reading progress toward the daily goal.", "parameters": {}},
    {"name": "listen_to_article", "description": "Listen to an article using text-to-speech.", "parameters": {"speed": {"type": "number", "description": "Optional playback speed multiplier (0.5 to 2.0).", "required": False}}},
]

POOL_ACCESSIBILITY = [
    {"name": "set_font_size", "description": "Set the system font size.", "parameters": {"size": {"type": "string", "description": "'small', 'medium', 'large', or 'extra_large'.", "required": True}}},
    {"name": "toggle_voice_assistant", "description": "Enable or disable the voice assistant listener.", "parameters": {"enabled": {"type": "boolean", "description": "True to enable, false to disable.", "required": True}}},
    {"name": "toggle_magnifier", "description": "Enable or disable the screen magnifier.", "parameters": {"enabled": {"type": "boolean", "description": "True to enable, false to disable.", "required": True}}},
    {"name": "toggle_screen_reader", "description": "Enable or disable the screen reader for accessibility.", "parameters": {"enabled": {"type": "boolean", "description": "True to enable, false to disable.", "required": True}}},
]

POOL_SHOPPING = [
    {"name": "search_product", "description": "Search for a product to buy online.", "parameters": {"query": {"type": "string", "description": "Product search query.", "required": True}}},
    {"name": "add_to_cart", "description": "Add a product to the shopping cart.", "parameters": {"product_name": {"type": "string", "description": "Name or description of the product.", "required": True}, "quantity": {"type": "number", "description": "Number of items to add.", "required": False}}},
    {"name": "check_order_status", "description": "Check the status of a recent order.", "parameters": {"order_id": {"type": "string", "description": "The order ID to check. If omitted, checks the most recent order.", "required": False}}},
]

POOL_SOCIAL = [
    {"name": "post_status_update", "description": "Post a status update or message to social media.", "parameters": {"text": {"type": "string", "description": "The status text to post.", "required": True}, "platform": {"type": "string", "description": "Social platform e.g. 'twitter', 'facebook', 'instagram'.", "required": False}}},
    {"name": "check_social_notifications", "description": "Check recent social media notifications.", "parameters": {"platform": {"type": "string", "description": "Optional platform filter.", "required": False}}},
    {"name": "like_post", "description": "Like a post on social media.", "parameters": {"post_id": {"type": "string", "description": "The ID of the post to like.", "required": True}}},
    {"name": "comment_on_post", "description": "Leave a comment on a social media post.", "parameters": {"post_id": {"type": "string", "description": "The ID of the post to comment on.", "required": True}, "text": {"type": "string", "description": "The comment text.", "required": True}}},
    {"name": "share_post", "description": "Share a post to another platform or feed.", "parameters": {"post_id": {"type": "string", "description": "The ID of the post to share.", "required": True}, "platform": {"type": "string", "description": "Optional target platform e.g. 'twitter', 'facebook'.", "required": False}}},
    {"name": "follow_user", "description": "Follow a user on social media.", "parameters": {"username": {"type": "string", "description": "The username to follow.", "required": True}}},
    {"name": "unfollow_user", "description": "Unfollow a user on social media.", "parameters": {"username": {"type": "string", "description": "The username to unfollow.", "required": True}}},
    {"name": "get_trending_topics", "description": "Get currently trending topics on social media.", "parameters": {"platform": {"type": "string", "description": "Optional platform filter.", "required": False}}},
    {"name": "send_direct_message", "description": "Send a direct message to a user on social media.", "parameters": {"username": {"type": "string", "description": "The recipient username.", "required": True}, "text": {"type": "string", "description": "The message text.", "required": True}}},
    {"name": "check_direct_messages", "description": "Check recent direct messages on social media.", "parameters": {"platform": {"type": "string", "description": "Optional platform filter.", "required": False}}},
]

POOL_RIDE_DELIVERY = [
    {"name": "request_ride", "description": "Request a ride to a destination.", "parameters": {"destination": {"type": "string", "description": "The destination address or place name.", "required": True}, "ride_type": {"type": "string", "description": "'standard', 'xl', or 'shared'.", "required": False}}},
    {"name": "cancel_ride", "description": "Cancel the current ride request.", "parameters": {}},
    {"name": "track_ride_eta", "description": "Track the ETA of the current ride.", "parameters": {}},
    {"name": "rate_ride", "description": "Rate a completed ride.", "parameters": {"rating": {"type": "number", "description": "Rating from 1 to 5.", "required": True}, "feedback": {"type": "string", "description": "Optional feedback text.", "required": False}}},
    {"name": "order_food", "description": "Order food delivery from a restaurant.", "parameters": {"restaurant": {"type": "string", "description": "Name of the restaurant.", "required": True}, "items": {"type": "string", "description": "Description of what to order.", "required": True}}},
    {"name": "track_food_delivery", "description": "Track the status of a food delivery order.", "parameters": {}},
]

POOL_FILE_MANAGEMENT = [
    {"name": "open_file", "description": "Open a file by name.", "parameters": {"file_name": {"type": "string", "description": "Name of the file to open.", "required": True}}},
    {"name": "share_file", "description": "Share a file with a contact.", "parameters": {"file_name": {"type": "string", "description": "Name of the file to share.", "required": True}, "contact_id": {"type": "string", "description": "The contact to share with.", "required": True}}},
    {"name": "download_file", "description": "Download a file from a URL.", "parameters": {"url": {"type": "string", "description": "The URL to download from.", "required": True}}},
    {"name": "move_file", "description": "Move a file to a different folder.", "parameters": {"file_name": {"type": "string", "description": "Name of the file to move.", "required": True}, "destination_folder": {"type": "string", "description": "The destination folder path.", "required": True}}},
    {"name": "compress_files", "description": "Compress files into an archive.", "parameters": {"file_names": {"type": "string", "description": "Description of files to compress.", "required": True}, "archive_name": {"type": "string", "description": "Name for the resulting archive.", "required": True}}},
    {"name": "create_folder", "description": "Create a new folder.", "parameters": {"folder_name": {"type": "string", "description": "Name of the folder to create.", "required": True}}},
    {"name": "delete_file", "description": "Delete a file.", "parameters": {"file_name": {"type": "string", "description": "Name of the file to delete.", "required": True}}},
]

POOL_WEARABLE = [
    {"name": "check_heart_rate", "description": "Check the current heart rate from the wearable sensor.", "parameters": {}},
    {"name": "check_blood_oxygen", "description": "Check the current blood oxygen level.", "parameters": {}},
    {"name": "start_breathing_exercise", "description": "Start a guided breathing exercise.", "parameters": {"duration_minutes": {"type": "number", "description": "Duration of the exercise in minutes.", "required": False}}},
    {"name": "find_my_phone", "description": "Trigger the phone to ring so you can find it.", "parameters": {}},
    {"name": "toggle_always_on_display", "description": "Enable or disable the always-on display.", "parameters": {"enabled": {"type": "boolean", "description": "True to enable, false to disable.", "required": True}}},
    {"name": "change_watch_face", "description": "Change the watch face.", "parameters": {"face_name": {"type": "string", "description": "Name of the watch face to switch to.", "required": True}}},
    {"name": "set_activity_goal", "description": "Set a daily activity goal.", "parameters": {"goal_type": {"type": "string", "description": "'steps', 'calories', or 'distance'.", "required": True}, "value": {"type": "number", "description": "The target value for the goal.", "required": True}}},
    {"name": "check_activity_progress", "description": "Check progress toward the current activity goal.", "parameters": {}},
    {"name": "toggle_fall_detection", "description": "Enable or disable fall detection.", "parameters": {"enabled": {"type": "boolean", "description": "True to enable, false to disable.", "required": True}}},
    {"name": "trigger_emergency_sos", "description": "Trigger an emergency SOS alert.", "parameters": {}},
]

POOL_DESKTOP = [
    {"name": "minimize_window", "description": "Minimize the current window.", "parameters": {}},
    {"name": "maximize_window", "description": "Maximize the current window.", "parameters": {}},
    {"name": "split_screen", "description": "Snap the current window to a screen position.", "parameters": {"position": {"type": "string", "description": "'left' or 'right'.", "required": True}}},
    {"name": "switch_virtual_desktop", "description": "Switch to another virtual desktop.", "parameters": {"direction": {"type": "string", "description": "'next', 'previous', or a desktop number.", "required": True}}},
    {"name": "toggle_vpn", "description": "Turn VPN on or off.", "parameters": {"enabled": {"type": "boolean", "description": "True to enable, false to disable.", "required": True}, "server": {"type": "string", "description": "Optional VPN server name or location.", "required": False}}},
    {"name": "switch_audio_output", "description": "Switch the audio output device.", "parameters": {"device_name": {"type": "string", "description": "Name of the output device e.g. 'External Speakers', 'HDMI'.", "required": True}}},
    {"name": "switch_audio_input", "description": "Switch the audio input device.", "parameters": {"device_name": {"type": "string", "description": "Name of the input device e.g. 'External Mic', 'Webcam'.", "required": True}}},
    {"name": "print_document", "description": "Print the current document.", "parameters": {"printer_name": {"type": "string", "description": "Optional printer name.", "required": False}, "copies": {"type": "number", "description": "Number of copies to print.", "required": False}}},
    {"name": "check_system_updates", "description": "Check for available system updates.", "parameters": {}},
    {"name": "install_system_updates", "description": "Install available system updates.", "parameters": {}},
    {"name": "take_screenshot_region", "description": "Take a screenshot of a specific region.", "parameters": {"region": {"type": "string", "description": "'full', 'window', or 'selection'.", "required": False}}},
    {"name": "kill_process", "description": "Force quit an application by name.", "parameters": {"app_name": {"type": "string", "description": "Name of the application to kill.", "required": True}}},
    {"name": "check_system_resources", "description": "Check current CPU, memory, and disk usage.", "parameters": {}},
]

POOL_BROWSER = [
    {"name": "open_tab", "description": "Open a new browser tab.", "parameters": {"url": {"type": "string", "description": "Optional URL to open.", "required": False}, "query": {"type": "string", "description": "Optional search query to open.", "required": False}}},
    {"name": "close_tab", "description": "Close the current browser tab.", "parameters": {}},
    {"name": "bookmark_page", "description": "Bookmark the current page.", "parameters": {}},
    {"name": "clear_browsing_data", "description": "Clear browsing data.", "parameters": {"data_type": {"type": "string", "description": "'history', 'cookies', 'cache', or 'all'.", "required": True}}},
    {"name": "open_private_window", "description": "Open a new private/incognito browser window.", "parameters": {}},
    {"name": "find_in_page", "description": "Find text on the current page.", "parameters": {"text": {"type": "string", "description": "The text to search for.", "required": True}}},
]

POOL_EMERGENCY_SAFETY = [
    {"name": "call_emergency_services", "description": "Call emergency services.", "parameters": {"service_type": {"type": "string", "description": "'police', 'fire', or 'ambulance'.", "required": False}}},
    {"name": "share_emergency_location", "description": "Share current location with an emergency contact.", "parameters": {"contact_id": {"type": "string", "description": "The emergency contact to share location with.", "required": True}}},
    {"name": "set_emergency_contact", "description": "Set up an emergency contact.", "parameters": {"name": {"type": "string", "description": "Full name of the emergency contact.", "required": True}, "phone": {"type": "string", "description": "Phone number of the emergency contact.", "required": True}}},
    {"name": "toggle_medical_id", "description": "Enable or disable the medical ID display.", "parameters": {"enabled": {"type": "boolean", "description": "True to enable, false to disable.", "required": True}}},
    {"name": "send_safety_check", "description": "Send a safety check-in to a contact.", "parameters": {"contact_id": {"type": "string", "description": "The contact to send the safety check to.", "required": True}, "message": {"type": "string", "description": "Optional message to include.", "required": False}}},
    {"name": "get_emergency_contacts", "description": "Get the list of emergency contacts.", "parameters": {}},
    {"name": "report_safety_issue", "description": "Report a safety issue or hazard.", "parameters": {"description": {"type": "string", "description": "Description of the safety issue.", "required": True}, "location": {"type": "string", "description": "Optional location of the issue.", "required": False}}},
    {"name": "enable_crash_detection", "description": "Enable or disable automatic crash detection.", "parameters": {"enabled": {"type": "boolean", "description": "True to enable, false to disable.", "required": True}}},
]

POOL_DIGITAL_WELLBEING = [
    {"name": "check_screen_time", "description": "Check screen time usage.", "parameters": {"period": {"type": "string", "description": "'today' or 'week'.", "required": False}}},
    {"name": "set_app_timer", "description": "Set a usage time limit for an app.", "parameters": {"app_name": {"type": "string", "description": "Name of the app to limit.", "required": True}, "duration_minutes": {"type": "number", "description": "Maximum usage time in minutes.", "required": True}}},
    {"name": "toggle_bedtime_mode", "description": "Enable or disable bedtime mode.", "parameters": {"enabled": {"type": "boolean", "description": "True to enable, false to disable.", "required": True}}},
    {"name": "set_content_restriction", "description": "Set content restriction level.", "parameters": {"restriction_level": {"type": "string", "description": "'off', 'moderate', or 'strict'.", "required": True}}},
    {"name": "check_app_usage", "description": "Check usage statistics for an app.", "parameters": {"app_name": {"type": "string", "description": "Optional app name to check. Shows all apps if omitted.", "required": False}}},
    {"name": "pause_all_notifications", "description": "Temporarily pause all notifications.", "parameters": {"duration_minutes": {"type": "number", "description": "Optional duration in minutes before resuming notifications.", "required": False}}},
    {"name": "get_daily_summary", "description": "Get a daily summary of device usage and wellbeing stats.", "parameters": {}},
    {"name": "set_downtime_schedule", "description": "Set a recurring downtime schedule to limit device use.", "parameters": {"start_time": {"type": "string", "description": "Start time for downtime e.g. '22:00'.", "required": True}, "end_time": {"type": "string", "description": "End time for downtime e.g. '07:00'.", "required": True}}},
]

POOL_TV_STREAMING = [
    {"name": "play_show", "description": "Play a TV show or movie by title on a streaming service.", "parameters": {"title": {"type": "string", "description": "Title of the show or movie.", "required": True}, "service": {"type": "string", "description": "Streaming service e.g. 'netflix', 'hulu', 'disney+', 'prime'.", "required": False}}},
    {"name": "switch_tv_input", "description": "Switch the TV input source.", "parameters": {"input": {"type": "string", "description": "Input source e.g. 'HDMI 1', 'HDMI 2', 'USB', 'antenna'.", "required": True}}},
    {"name": "toggle_subtitles", "description": "Turn subtitles on or off.", "parameters": {"enabled": {"type": "boolean", "description": "True to enable, false to disable.", "required": True}, "language": {"type": "string", "description": "Subtitle language e.g. 'english', 'spanish'.", "required": False}}},
    {"name": "set_tv_volume", "description": "Set the TV volume level.", "parameters": {"level": {"type": "number", "description": "Volume level from 0 to 100.", "required": True}}},
    {"name": "tv_power", "description": "Turn the TV on or off.", "parameters": {"action": {"type": "string", "description": "'on' or 'off'.", "required": True}}},
    {"name": "tv_channel", "description": "Switch to a specific TV channel.", "parameters": {"channel": {"type": "string", "description": "Channel number or name.", "required": True}}},
    {"name": "rewind_media", "description": "Rewind the current media by a specified amount.", "parameters": {"seconds": {"type": "number", "description": "Number of seconds to rewind.", "required": False}}},
    {"name": "fast_forward_media", "description": "Fast forward the current media by a specified amount.", "parameters": {"seconds": {"type": "number", "description": "Number of seconds to fast forward.", "required": False}}},
    {"name": "search_streaming", "description": "Search for content across streaming services.", "parameters": {"query": {"type": "string", "description": "Search query for movies, shows, or actors.", "required": True}}},
    {"name": "add_to_watchlist", "description": "Add a show or movie to your watchlist.", "parameters": {"title": {"type": "string", "description": "Title to add.", "required": True}}},
    {"name": "get_continue_watching", "description": "Get a list of shows/movies you haven't finished.", "parameters": {}},
    {"name": "set_sleep_timer_tv", "description": "Set a sleep timer to auto-turn off the TV.", "parameters": {"minutes": {"type": "number", "description": "Minutes until the TV turns off.", "required": True}}},
]

POOL_VEHICLE = [
    {"name": "lock_car", "description": "Lock or unlock the car remotely.", "parameters": {"action": {"type": "string", "description": "'lock' or 'unlock'.", "required": True}}},
    {"name": "start_car", "description": "Remote start or stop the car engine.", "parameters": {"action": {"type": "string", "description": "'start' or 'stop'.", "required": True}}},
    {"name": "car_climate", "description": "Set the car's climate control remotely.", "parameters": {"temperature": {"type": "number", "description": "Target temperature in degrees.", "required": True}, "unit": {"type": "string", "description": "'fahrenheit' or 'celsius'.", "required": False}}},
    {"name": "find_my_car", "description": "Locate your parked car on a map.", "parameters": {}},
    {"name": "check_car_status", "description": "Check car status: fuel/battery level, tire pressure, doors.", "parameters": {}},
    {"name": "honk_horn", "description": "Honk the car horn remotely to locate it.", "parameters": {}},
    {"name": "flash_lights", "description": "Flash the car lights remotely.", "parameters": {}},
    {"name": "open_trunk", "description": "Open or close the car trunk remotely.", "parameters": {"action": {"type": "string", "description": "'open' or 'close'.", "required": True}}},
    {"name": "get_fuel_level", "description": "Check the current fuel or battery level.", "parameters": {}},
    {"name": "find_ev_charger", "description": "Find nearby EV charging stations.", "parameters": {"connector_type": {"type": "string", "description": "Optional connector type e.g. 'CCS', 'CHAdeMO', 'Tesla'.", "required": False}}},
    {"name": "set_charge_limit", "description": "Set the EV charge limit percentage.", "parameters": {"percent": {"type": "number", "description": "Charge limit 50-100%.", "required": True}}},
    {"name": "get_range", "description": "Get the estimated remaining driving range.", "parameters": {}},
    {"name": "toggle_valet_mode", "description": "Enable or disable valet mode.", "parameters": {"enabled": {"type": "boolean", "description": "True to enable, false to disable.", "required": True}}},
]

POOL_TRAVEL = [
    {"name": "search_flights", "description": "Search for flights between cities.", "parameters": {"origin": {"type": "string", "description": "Origin city or airport code.", "required": True}, "destination": {"type": "string", "description": "Destination city or airport code.", "required": True}, "date": {"type": "string", "description": "Travel date in human readable format.", "required": True}}},
    {"name": "search_hotels", "description": "Search for hotels in a location.", "parameters": {"location": {"type": "string", "description": "City or area to search.", "required": True}, "check_in": {"type": "string", "description": "Check-in date.", "required": True}, "check_out": {"type": "string", "description": "Check-out date.", "required": True}}},
    {"name": "check_in_flight", "description": "Check in for an upcoming flight.", "parameters": {"confirmation_code": {"type": "string", "description": "Booking confirmation code.", "required": True}}},
    {"name": "show_boarding_pass", "description": "Display the boarding pass for an upcoming flight.", "parameters": {"confirmation_code": {"type": "string", "description": "Optional booking confirmation code. Shows next flight if omitted.", "required": False}}},
    {"name": "get_flight_status", "description": "Check the status of a flight.", "parameters": {"flight_number": {"type": "string", "description": "Flight number e.g. 'UA123', 'DL456'.", "required": True}}},
    {"name": "book_rental_car", "description": "Search for rental cars at a location.", "parameters": {"location": {"type": "string", "description": "Pickup location.", "required": True}, "pickup_date": {"type": "string", "description": "Pickup date.", "required": True}, "return_date": {"type": "string", "description": "Return date.", "required": True}}},
    {"name": "get_trip_itinerary", "description": "Show the itinerary for an upcoming trip.", "parameters": {"trip_name": {"type": "string", "description": "Optional trip name or destination.", "required": False}}},
    {"name": "convert_timezone", "description": "Convert a time between timezones.", "parameters": {"time": {"type": "string", "description": "Time to convert e.g. '3pm'.", "required": True}, "from_tz": {"type": "string", "description": "Source timezone e.g. 'EST', 'America/New_York'.", "required": True}, "to_tz": {"type": "string", "description": "Target timezone.", "required": True}}},
    {"name": "get_exchange_rate", "description": "Get the current exchange rate between currencies.", "parameters": {"from_currency": {"type": "string", "description": "Source currency code.", "required": True}, "to_currency": {"type": "string", "description": "Target currency code.", "required": True}}},
    {"name": "translate_phrase", "description": "Translate a phrase for travel purposes.", "parameters": {"text": {"type": "string", "description": "Text to translate.", "required": True}, "target_language": {"type": "string", "description": "Target language.", "required": True}}},
]

POOL_COOKING_KITCHEN = [
    {"name": "search_recipe", "description": "Search for a recipe by dish name or ingredients.", "parameters": {"query": {"type": "string", "description": "Dish name or ingredients to search for.", "required": True}}},
    {"name": "set_oven", "description": "Preheat or set the smart oven temperature.", "parameters": {"temperature": {"type": "number", "description": "Temperature in degrees.", "required": True}, "unit": {"type": "string", "description": "'fahrenheit' or 'celsius'.", "required": False}, "mode": {"type": "string", "description": "'bake', 'broil', 'convection', or 'air_fry'.", "required": False}}},
    {"name": "start_dishwasher", "description": "Start or schedule the dishwasher.", "parameters": {"cycle": {"type": "string", "description": "'normal', 'heavy', 'quick', or 'eco'.", "required": False}, "delay_minutes": {"type": "number", "description": "Optional delay before starting.", "required": False}}},
    {"name": "control_coffee_maker", "description": "Control the smart coffee maker.", "parameters": {"action": {"type": "string", "description": "'brew', 'stop', or 'warm'.", "required": True}, "size": {"type": "string", "description": "'small', 'medium', or 'large'.", "required": False}, "strength": {"type": "string", "description": "'mild', 'medium', or 'strong'.", "required": False}}},
    {"name": "cooking_convert", "description": "Convert between cooking measurements.", "parameters": {"value": {"type": "number", "description": "The value to convert.", "required": True}, "from_unit": {"type": "string", "description": "Source unit e.g. 'cups', 'tablespoons', 'grams', 'ounces'.", "required": True}, "to_unit": {"type": "string", "description": "Target unit.", "required": True}}},
    {"name": "set_cooking_timer", "description": "Set a named cooking timer.", "parameters": {"label": {"type": "string", "description": "Label for the timer e.g. 'pasta', 'chicken', 'rice'.", "required": True}, "minutes": {"type": "number", "description": "Timer duration in minutes.", "required": True}}},
    {"name": "get_substitution", "description": "Get a cooking ingredient substitution.", "parameters": {"ingredient": {"type": "string", "description": "The ingredient to substitute e.g. 'buttermilk', 'eggs'.", "required": True}}},
    {"name": "control_slow_cooker", "description": "Control the smart slow cooker.", "parameters": {"action": {"type": "string", "description": "'start', 'stop', or 'warm'.", "required": True}, "setting": {"type": "string", "description": "'low', 'medium', or 'high'.", "required": False}, "hours": {"type": "number", "description": "Cooking duration in hours.", "required": False}}},
    {"name": "check_fridge_inventory", "description": "Check what's in the smart fridge.", "parameters": {"category": {"type": "string", "description": "Optional category e.g. 'dairy', 'vegetables', 'drinks'.", "required": False}}},
]

POOL_PARENTAL_FAMILY = [
    {"name": "get_family_location", "description": "Check the current location of a family member.", "parameters": {"member": {"type": "string", "description": "Name of the family member.", "required": True}}},
    {"name": "set_child_screen_time", "description": "Set a daily screen time limit for a child's device.", "parameters": {"child_name": {"type": "string", "description": "Name of the child.", "required": True}, "minutes": {"type": "number", "description": "Daily screen time limit in minutes.", "required": True}}},
    {"name": "lock_child_device", "description": "Immediately lock or unlock a child's device.", "parameters": {"child_name": {"type": "string", "description": "Name of the child.", "required": True}, "action": {"type": "string", "description": "'lock' or 'unlock'.", "required": True}}},
    {"name": "get_child_activity", "description": "Get a child's recent device activity and app usage.", "parameters": {"child_name": {"type": "string", "description": "Name of the child.", "required": True}}},
    {"name": "set_content_filter", "description": "Set content filtering level for a child's device.", "parameters": {"child_name": {"type": "string", "description": "Name of the child.", "required": True}, "level": {"type": "string", "description": "'strict', 'moderate', or 'off'.", "required": True}}},
    {"name": "send_family_broadcast", "description": "Send a message to all family members.", "parameters": {"message": {"type": "string", "description": "The message to broadcast.", "required": True}}},
    {"name": "set_family_geofence", "description": "Set an alert when a family member arrives at or leaves a location.", "parameters": {"member": {"type": "string", "description": "Family member name.", "required": True}, "location": {"type": "string", "description": "Location name e.g. 'school', 'home', 'work'.", "required": True}, "trigger": {"type": "string", "description": "'arrive' or 'leave'.", "required": True}}},
    {"name": "request_check_in", "description": "Send a check-in request to a family member.", "parameters": {"member": {"type": "string", "description": "Family member name.", "required": True}}},
]

POOL_AUTOMATION = [
    {"name": "run_shortcut", "description": "Run a saved automation shortcut by name.", "parameters": {"name": {"type": "string", "description": "Name of the shortcut e.g. 'morning routine', 'bedtime', 'leaving home'.", "required": True}}},
    {"name": "create_routine", "description": "Create a new automation routine with a trigger and actions.", "parameters": {"name": {"type": "string", "description": "Name for the routine.", "required": True}, "trigger": {"type": "string", "description": "When to run e.g. 'every weekday at 7am', 'when I arrive home', 'at sunset'.", "required": True}, "actions": {"type": "string", "description": "Description of actions to perform.", "required": True}}},
    {"name": "list_routines", "description": "List all saved automation routines.", "parameters": {}},
    {"name": "delete_routine", "description": "Delete an automation routine by name.", "parameters": {"name": {"type": "string", "description": "Name of the routine to delete.", "required": True}}},
    {"name": "toggle_routine", "description": "Enable or disable an automation routine.", "parameters": {"name": {"type": "string", "description": "Name of the routine.", "required": True}, "enabled": {"type": "boolean", "description": "True to enable, false to disable.", "required": True}}},
    {"name": "run_scene", "description": "Activate a smart home scene.", "parameters": {"scene_name": {"type": "string", "description": "Scene name e.g. 'movie night', 'dinner party', 'good morning', 'away'.", "required": True}}},
    {"name": "get_routine_history", "description": "Check when a routine last ran.", "parameters": {"name": {"type": "string", "description": "Name of the routine.", "required": True}}},
]

POOL_CONNECTIVITY = [
    {"name": "toggle_mobile_data", "description": "Turn mobile data on or off.", "parameters": {"enabled": {"type": "boolean", "description": "True to enable, false to disable.", "required": True}}},
    {"name": "toggle_nfc", "description": "Turn NFC on or off.", "parameters": {"enabled": {"type": "boolean", "description": "True to enable, false to disable.", "required": True}}},
    {"name": "share_wifi", "description": "Share the current WiFi network credentials via QR code or nearby share.", "parameters": {}},
    {"name": "get_data_usage", "description": "Check mobile data usage for the current billing period.", "parameters": {}},
    {"name": "set_data_limit", "description": "Set a mobile data usage warning or limit.", "parameters": {"limit_gb": {"type": "number", "description": "Data limit in gigabytes.", "required": True}}},
    {"name": "get_wifi_info", "description": "Get info about the current WiFi connection: network name, signal strength, speed.", "parameters": {}},
    {"name": "forget_wifi_network", "description": "Forget a saved WiFi network.", "parameters": {"ssid": {"type": "string", "description": "Name of the WiFi network to forget.", "required": True}}},
    {"name": "get_connected_devices", "description": "List all Bluetooth or WiFi devices currently connected.", "parameters": {"connection_type": {"type": "string", "description": "'bluetooth' or 'wifi'.", "required": False}}},
    {"name": "disconnect_bluetooth_device", "description": "Disconnect a specific Bluetooth device.", "parameters": {"device_name": {"type": "string", "description": "Name of the device to disconnect.", "required": True}}},
]

POOL_SCREEN_CAPTURE = [
    {"name": "start_screen_recording", "description": "Start recording the screen.", "parameters": {"audio": {"type": "boolean", "description": "Whether to record microphone audio too.", "required": False}}},
    {"name": "stop_screen_recording", "description": "Stop the current screen recording and save it.", "parameters": {}},
    {"name": "take_scrolling_screenshot", "description": "Take a long scrolling screenshot of the current page.", "parameters": {}},
    {"name": "annotate_screenshot", "description": "Open the most recent screenshot for annotation/markup.", "parameters": {}},
    {"name": "screen_mirror", "description": "Mirror the screen to an external display.", "parameters": {"device_name": {"type": "string", "description": "Name of the display e.g. 'Living Room TV', 'Office Monitor'.", "required": True}}},
    {"name": "stop_screen_mirror", "description": "Stop screen mirroring.", "parameters": {}},
    {"name": "picture_in_picture", "description": "Enable or disable picture-in-picture mode for the current app.", "parameters": {"enabled": {"type": "boolean", "description": "True to enable, false to disable.", "required": True}}},
]

POOL_EMAIL = [
    {"name": "compose_email", "description": "Compose and send an email.", "parameters": {"to": {"type": "string", "description": "Recipient email address.", "required": True}, "subject": {"type": "string", "description": "Email subject line.", "required": True}, "body": {"type": "string", "description": "Email body text.", "required": True}, "cc": {"type": "string", "description": "Optional CC recipients.", "required": False}}},
    {"name": "reply_email", "description": "Reply to the most recent or specified email.", "parameters": {"body": {"type": "string", "description": "Reply text.", "required": True}, "subject_match": {"type": "string", "description": "Optional subject line to identify which email to reply to.", "required": False}}},
    {"name": "forward_email", "description": "Forward an email to someone.", "parameters": {"to": {"type": "string", "description": "Recipient to forward to.", "required": True}, "subject_match": {"type": "string", "description": "Subject line to identify the email.", "required": False}}},
    {"name": "archive_email", "description": "Archive an email.", "parameters": {"subject_match": {"type": "string", "description": "Subject line to identify the email.", "required": True}}},
    {"name": "snooze_email", "description": "Snooze an email to reappear later.", "parameters": {"subject_match": {"type": "string", "description": "Subject line to identify the email.", "required": True}, "snooze_until": {"type": "string", "description": "When to resurface e.g. 'tomorrow morning', 'next Monday'.", "required": True}}},
    {"name": "label_email", "description": "Apply a label or folder to an email.", "parameters": {"subject_match": {"type": "string", "description": "Subject to identify the email.", "required": True}, "label": {"type": "string", "description": "Label name e.g. 'work', 'personal', 'receipts'.", "required": True}}},
    {"name": "mark_email_spam", "description": "Mark an email as spam.", "parameters": {"subject_match": {"type": "string", "description": "Subject to identify the email.", "required": True}}},
    {"name": "get_unread_emails", "description": "Get unread emails, optionally filtered.", "parameters": {"from_contact": {"type": "string", "description": "Optional sender name to filter by.", "required": False}, "count": {"type": "number", "description": "Number of emails to show.", "required": False}}},
    {"name": "search_email", "description": "Search emails by keyword.", "parameters": {"query": {"type": "string", "description": "Search query.", "required": True}}},
]

POOL_PASSWORDS = [
    {"name": "get_password", "description": "Retrieve a saved password for a site or app.", "parameters": {"site": {"type": "string", "description": "Website or app name.", "required": True}}},
    {"name": "generate_password", "description": "Generate a secure random password.", "parameters": {"length": {"type": "number", "description": "Password length (default 16).", "required": False}, "include_symbols": {"type": "boolean", "description": "Whether to include symbols.", "required": False}}},
    {"name": "get_totp_code", "description": "Get the current 2FA/TOTP code for a site.", "parameters": {"site": {"type": "string", "description": "Website or app name.", "required": True}}},
    {"name": "save_password", "description": "Save or update a password for a site.", "parameters": {"site": {"type": "string", "description": "Website or app name.", "required": True}, "username": {"type": "string", "description": "Username or email.", "required": True}, "password": {"type": "string", "description": "The password to save.", "required": True}}},
    {"name": "check_password_breach", "description": "Check if a password or account has been in a data breach.", "parameters": {"site": {"type": "string", "description": "Optional site to check. Checks all if omitted.", "required": False}}},
    {"name": "autofill_login", "description": "Autofill login credentials on the current page.", "parameters": {}},
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
    POOL_RIDE_DELIVERY,
    POOL_FILE_MANAGEMENT,
    POOL_WEARABLE,
    POOL_DESKTOP,
    POOL_BROWSER,
    POOL_EMERGENCY_SAFETY,
    POOL_DIGITAL_WELLBEING,
    POOL_TV_STREAMING,
    POOL_VEHICLE,
    POOL_TRAVEL,
    POOL_COOKING_KITCHEN,
    POOL_PARENTAL_FAMILY,
    POOL_AUTOMATION,
    POOL_CONNECTIVITY,
    POOL_SCREEN_CAPTURE,
    POOL_EMAIL,
    POOL_PASSWORDS,
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
    "posting on social media", "checking social notifications",
    "posting a status update",
    # Ride-hailing & delivery
    "ordering a ride to a destination", "tracking a food delivery",
    "canceling a ride", "ordering food from a nearby place",
    # File management
    "opening a downloaded PDF", "sharing a document with a coworker",
    "creating a new folder for photos", "compressing files to send",
    "moving a file to the downloads folder",
    # Wearable
    "checking heart rate during exercise", "finding my phone from my watch",
    "changing the watch face", "starting a breathing exercise",
    "checking if I hit my step goal", "checking blood oxygen before sleep",
    # Desktop/laptop
    "splitting the screen for multitasking", "switching to external speakers",
    "printing a document", "killing a frozen app",
    "connecting to VPN for work", "checking for system updates",
    "switching to the next virtual desktop",
    # Browser
    "opening a new private window", "bookmarking this page",
    "clearing browser history", "finding text on this page",
    # Emergency & safety
    "calling 911", "sending my location to emergency contact",
    "triggering SOS from watch", "setting up emergency contacts",
    # Digital wellbeing
    "checking how much time I spent on my phone today",
    "setting a 30 minute limit on social media",
    "turning on bedtime mode", "checking which app I used most this week",
    # Cross-device & undo
    "cancel the timer I just set", "undo the last action",
    "reply to the last message", "call back the last number",
    "find the nearest pharmacy and navigate there",
    # Meeting & productivity context
    "noting an action item from a meeting", "scheduling a follow-up meeting with the team",
    "reminding me to send the proposal after the call", "texting someone I'm running late to the meeting",
    "setting a timer for a 15 minute break between meetings", "adding meeting notes about the budget discussion",
    "creating a todo for the deliverables we agreed on", "sharing meeting notes with attendees",
    "scheduling a 1-on-1 for next week", "setting do not disturb for an hour-long meeting",
    "recording a voice memo of key takeaways", "emailing the client a summary after the call",
    "checking my calendar for conflicts before accepting", "creating a checklist of action items from standup",
    # Indirect / implicit intent — observations, feelings, complaints (not commands)
    "it's really dark in here", "I can't hear anything on this call",
    "I'm freezing", "it's too bright", "I'm bored",
    "I need to wake up early tomorrow", "I keep forgetting to drink water",
    "my phone is about to die", "I can barely see my screen",
    "this room is stuffy", "it's so loud in here", "I can't sleep",
    "I'm lost and I don't know where I am", "my eyes are hurting from this screen",
    "it's getting dark outside", "I'm starving", "I wonder what the weather's like",
    "ugh I have so much to do today", "I'm late for my meeting",
    "my back hurts from sitting all day", "I haven't moved in hours",
    "it smells like something is burning in the kitchen", "the house feels empty and creepy",
    "I'm so tired", "why is it so hot in here", "the baby is sleeping so everything needs to be quiet",
    "I think I left the front door unlocked", "my hands are full and I need light",
    "this song is terrible", "I have no idea what time it is over there",
    "I promised someone I'd call them back", "the kids have been on their tablets all day",
    "I should probably stretch or something", "I wonder if my package arrived",
    "this article is really long I don't have time to read all of it",
    "I need to remember to buy milk on the way home",
    "my car must be low on gas by now", "I have no idea where I parked",
    # Corrections and follow-ups
    "correcting a time just set", "redirecting a message to a different person",
    "cancel what I just set", "change the timer to 10 minutes",
    "undo that", "never mind, turn it back on",
    # Temporal and conditional
    "every weekday at 8am", "when I get home remind me to call mom",
    "in 2 hours turn off the lights", "tomorrow morning check the weather",
    "after work start the robot vacuum", "at sunset close the blinds",
    # Multi-person and group
    "messaging two people about the same event", "setting up a group call",
    "sharing location with a family group",
    "sending the same email to multiple recipients",
    # Context references
    "call them back", "open that file from earlier",
    "reply saying I'll be there in 10", "send them what I just screenshotted",
    "play that song again", "go back to the last page",
    # Multilingual and code-switched
    "set un timer por 5 minutos", "envoie un message à Jean",
    "spiel etwas Musik", "mets le volume à 50",
    "llama a mamá", "recherche le restaurant le plus proche",
    # Ultra-terse commands
    "timer 5", "alarm 7am", "lights off", "call mom", "weather",
    "screenshot", "wifi off", "volume up", "next song", "lock door",
    # Verbose and conversational
    "hey so I was thinking could you maybe set a reminder for me to pick up the dry cleaning sometime tomorrow afternoon if that's not too much trouble",
    "ok so I need you to do a few things: first turn off the living room lights, then lock the front door, and also set the alarm for 6:30",
    "I'm about to go into a meeting so can you put my phone on do not disturb and also remind me in an hour to check my email",
    # Multi-call with same tool (few tools, different args)
    "sending different messages to two people about different topics",
    "setting two alarms for different times and purposes",
    "adding multiple grocery items to a shopping list separately",
    "creating reminders for several appointments across the week",
    "texting three different people with different status updates",
    "setting multiple cooking timers for different dishes",
    "sending two separate emails about different work topics",
    "adding several distinct notes about different subjects",
    "creating multiple calendar events for different days",
    "logging several meals throughout the day",
    # Multi-call with long argument values
    "sending a detailed work message with a specific reason for a schedule change",
    "creating a note with a full recipe including all ingredients and steps",
    "emailing someone about a maintenance issue with a detailed explanation",
    "setting a reminder that includes a full street address",
    "posting a long enthusiastic status update about an upcoming event",
    # TV & streaming
    "playing a show on a streaming service", "turning on subtitles", "switching TV input",
    "setting a TV sleep timer", "searching for a genre of movies", "rewinding playback",
    "resuming what was playing last", "adding something to a watchlist",
    "adjusting TV volume to a specific level", "playing the next episode",
    # Vehicle & car
    "locking the car remotely", "remote starting the car with climate on", "finding where I parked",
    "checking tire pressure", "honking to locate the car",
    "checking fuel or battery level", "preheating the cabin",
    "opening the trunk remotely", "finding a charging station", "setting EV charge limit",
    "checking if car is locked", "enabling valet mode",
    # Travel & booking
    "searching for flights to a destination", "checking in for a flight",
    "showing a boarding pass", "checking flight status", "checking currency exchange rate",
    "checking time in another timezone", "searching for hotels in a city",
    "viewing a trip itinerary", "renting a car at an airport",
    # Cooking & kitchen
    "preheating the oven to a temperature", "converting cooking measurements",
    "setting a named cooking timer", "finding an ingredient substitution",
    "starting the dishwasher on a specific cycle", "brewing coffee with size and strength",
    "searching for a specific recipe", "checking fridge inventory",
    "starting the slow cooker with settings",
    # Parental & family
    "checking a family member's location", "locking a child's device",
    "checking a child's screen time usage", "setting screen time limits",
    "broadcasting a message to the whole family",
    "geofencing alerts for when someone arrives or leaves", "requesting a check-in",
    "setting content filters on kids' devices",
    # Automation & shortcuts
    "run my morning routine", "trigger the bedtime shortcut",
    "what automations do I have", "turn off the leaving home routine",
    "run movie night scene", "delete the weekend cleanup routine",
    "when did my morning routine last run",
    "create a routine that turns off lights at 11pm every night",
    # Connectivity
    "turn off mobile data", "how much data have I used this month",
    "what wifi am I connected to", "share the wifi password",
    "forgetting a saved wifi network", "listing connected bluetooth devices",
    "disconnecting a specific bluetooth device", "toggling NFC", "setting a data usage limit",
    # Screen capture & mirroring
    "start screen recording", "stop recording the screen",
    "take a scrolling screenshot", "mirror my screen to the living room TV",
    "stop mirroring", "turn on picture in picture",
    "record the screen with my mic on",
    # Email management
    "checking unread emails", "replying to an email from someone",
    "archiving an email about a topic", "snoozing an email until later",
    "labeling an email by category", "marking email as spam",
    "forwarding an email to someone", "searching emails by keyword",
    "checking emails from a specific sender",
    # Passwords & security
    "retrieving a saved password for a service", "generating a new password",
    "getting a 2FA code for a site", "checking for data breaches",
    "autofilling login credentials", "saving a new password",
    # Chained real-world workflows
    "I just parked — save this location and set a 2 hour timer",
    "I'm leaving work — start the car, turn on the AC, and get directions home",
    "going to bed — lock the doors, turn off all lights, and set alarm for 6:30",
    "heading to a meeting — put phone on DND, text my wife I'll be late, and set a 1 hour timer",
    "at the airport — check in for my flight, show boarding pass, and turn on airplane mode",
    "cooking dinner — preheat oven to 400, set a timer for chicken 45 minutes, and play jazz",
    # Device-state queries
    "is my bluetooth on", "what devices are connected", "how much storage do I have left",
    "what's my battery percentage", "is wifi connected", "what network am I on",
    "is do not disturb on", "what's the screen brightness set to",
    # Quantified and precise requests
    "set brightness to exactly 40 percent", "volume to 30", "timer for 7 minutes and 30 seconds",
    "set thermostat to 68.5 degrees", "alarm for 6:15am sharp",
    "rewind exactly 10 seconds", "set charge limit to 90 percent",
    # Negation and cancellation
    "don't set an alarm for tomorrow", "stop reminding me about the dentist",
    "I don't want notifications from Instagram anymore",
    "turn off all my alarms", "remove everything from my shopping list",
    "unsubscribe from the sports newsletter", "disable the morning routine",
    "cancel my ride", "stop the screen recording",
    # Disfluent / natural speech — filler words, false starts, self-corrections
    "um can you like set a timer for uh 10 minutes",
    "hey so I was thinking like maybe turn off the lights or whatever",
    "okay so uh send a message to... wait who was it... oh right, send it to Mike",
    "alright so um I need an alarm for like 7... no wait 7:30 tomorrow morning",
    "hmm can you uh play some music, like jazz or something chill I guess",
    "oh yeah hey turn the volume down a bit it's kinda loud",
    "so like I think I need to uh check my email or something",
    "wait actually no, set it to 20 minutes not 15",
    "could you um like maybe turn on do not disturb if that's a thing",
    "hey uh what's the... what's the weather like out there",
    "rambling about needing to text someone that you're running late",
    "set a — no actually make that a reminder, not a timer, for 3pm",
    "uh brightness, like turn it up, I can't really see",
    "so I think maybe we should lock the front door, right? yeah do that",
    "hey siri — I mean, uh, can you just play the next song",
    "well I guess I should probably set an alarm, like for early, maybe 6?",
    "turn turn off the uh the bluetooth thing",
    "oh um could you check like how much battery I have left or whatever",
    "right so basically I need directions to the uh the airport, yeah",
    "let me think... okay yeah just add milk to the shopping list",
    "so um like my mom asked me to call her so yeah can you call mom",
    "hey wait before I forget can you remind me tomorrow about the dentist thing",
    "I dunno just like play something, anything really, some background music",
    "ugh okay fine just turn on the flashlight I can't see anything",
    # Typos and keyboard errors — themes for Gemini to generate its own misspellings
    "sending a message with typos in the command",
    "setting a timer with misspelled words",
    "playing music with garbled text", "toggling a setting with missing letters",
    "checking something with swapped letters", "calling someone with truncated words",
    "searching nearby with typos", "toggling connectivity with misspelling",
    "checking email with keyboard errors", "locking something with missing vowels",
    "running words together without spaces", "setting something with no spaces",
    "adjusting a setting with a typo", "navigating with misspelled destination",
    "a reminder with text-speak abbreviations", "texting with informal shorthand",
    # Speech-to-text / ASR errors — themes
    "a timer request where a number sounds like a different word",
    "connecting a device where the name gets mangled by speech recognition",
    "playing music where the genre gets misheard",
    "checking something where homophones get confused (weather/whether, to/two, for/four)",
    "casual speech with wanna/gonna/gotta contractions",
    "slang and phonetic spelling like lemme/gimme/hafta/shoulda",
    # Dropped words / telegram style — themes
    "ultra-short command with just a number or single word",
    "device command with just the setting name and value, no verbs",
    "message with just recipient and content, no filler words",
    "directions with just destination and constraint, no articles",
    "reminder with just topic, day, and time — no complete sentence",
    "multiple toggles in telegram style, no connecting words",
    # Ambiguous queries requiring disambiguation between similar tools
    "sending something to someone without specifying the channel",
    "messaging someone about a topic without specifying SMS vs instant message",
    "setting a reminder without specifying reminder vs list item",
    "playing something without specifying song vs playlist vs radio",
    "navigating somewhere without specifying directions vs turn-by-turn",
    "setting a timer without specifying generic vs cooking timer",
    "writing something down without specifying note vs list vs reminder",
    "sharing something without specifying the method or recipient type",
    "looking up a place without specifying search vs navigate",
    "telling someone something without specifying call vs text vs email",
    "booking something without specifying the service type",
    # Argument value edge cases — themes for special characters, diverse content
    "texting someone with emoji in the message",
    "emailing someone at a non-English domain about a work topic",
    "setting a reminder with a full street address including apartment number",
    "noting down a complex password or code with special characters",
    "sending a casual excited message with lots of emoji",
    "emailing a colleague at a foreign company about a deadline",
    "creating a note with a recipe using fractions and degree symbols",
    "texting someone about travel logistics with flight details",
    "sending a professional email with specific numbers, percentages, and a meeting time",
    "adding a shopping list with quantities, brands, and metric measurements",
    # Math, calculations, and conversions
    "what's 15 percent of 89.99", "split 247 dollars three ways",
    "how many days until March 15", "what's 72 fahrenheit in celsius",
    "calculate the area of a circle with radius 5", "what's the square root of 144",
    "how much is 30% off of $65", "if I drive 180 miles at 60mph how long will it take",
    "convert 2.5 hours to minutes and seconds", "what's 19 times 23 plus 7",
    "how many calories if I eat 3 servings of 240 calories each",
    "what's my average if I scored 87, 92, 78, and 95", "tip for $43.50 at 20%",
    "what day of the week is January 1 2025", "BMI for 175 lbs and 5 foot 10",
    "how many cups in 750ml", "convert 3 tablespoons to teaspoons",
    "what's 1000 yen in dollars roughly", "compound interest on 5000 at 4% for 3 years",
    # Edge cases
    "very short terse command like 'timer 5 min'",
    "long polite request with extra context",
    "ambiguous request that maps to one tool",
    "casual voice assistant style like 'hey set my alarm for seven'",
    "request where no available tool can help",
]

CALL_TYPES = [
    ("single", "exactly 1 tool call"),                                                                      # 3/16 ≈ 19%
    ("single", "exactly 1 tool call"),
    ("single", "exactly 1 tool call"),
    ("multi", "2-3 tool calls (the user wants multiple things done at once)"),                               # 3/16 ≈ 19%
    ("multi", "2-3 tool calls (the user wants multiple things done at once)"),
    ("multi", "2-3 tool calls (the user wants multiple things done at once)"),
    ("multi_few_tools", "2-4 tool calls using ONLY 1-3 of the available tools — the user wants multiple actions with the SAME or very few tools, with DIFFERENT detailed argument values each time (e.g. sending messages to 3 different people, setting 2 different alarms, creating multiple list items). Each call MUST have distinct, realistic argument values — vary names, times, locations, messages, etc."),  # 4/16 ≈ 25%
    ("multi_few_tools", "2-4 tool calls using ONLY 1-3 of the available tools — the user wants multiple actions with the SAME or very few tools, with DIFFERENT detailed argument values each time (e.g. sending messages to 3 different people, setting 2 different alarms, creating multiple list items). Each call MUST have distinct, realistic argument values — vary names, times, locations, messages, etc."),
    ("multi_few_tools", "2-4 tool calls using ONLY 1-3 of the available tools — the user wants multiple actions with the SAME or very few tools, with DIFFERENT detailed argument values each time (e.g. sending messages to 3 different people, setting 2 different alarms, creating multiple list items). Each call MUST have distinct, realistic argument values — vary names, times, locations, messages, etc."),
    ("multi_few_tools", "2-4 tool calls using ONLY 1-3 of the available tools — the user wants multiple actions with the SAME or very few tools, with DIFFERENT detailed argument values each time (e.g. sending messages to 3 different people, setting 2 different alarms, creating multiple list items). Each call MUST have distinct, realistic argument values — vary names, times, locations, messages, etc."),
    ("multi_long_values", "2-3 tool calls where argument values are LONG and detailed — full sentences for message text, multi-word descriptions, specific addresses, complete email bodies, detailed notes. Each argument value should be at least 5-10 words, not just a single word or number."),  # 2/16 ≈ 12%
    ("multi_long_values", "2-3 tool calls where argument values are LONG and detailed — full sentences for message text, multi-word descriptions, specific addresses, complete email bodies, detailed notes. Each argument value should be at least 5-10 words, not just a single word or number."),
    ("none", "NO tool calls — the user asks a question or makes a request that NONE of the available tools can fulfill. "
     "answers must be []. Every query must be specific, detailed, and UNIQUE — no generic one-liners."),
    ("near_miss", "NO tool calls — the query is RELATED to the domain of the available tools but the tools CANNOT actually fulfill the request. "
     "Examples: asking for HISTORY of alarms when only set_alarm exists, asking to EDIT a calendar event when only create/delete exist, "
     "asking for a feature a tool doesn't support (e.g. 'set a recurring timer' when set_timer has no repeat param), "
     "asking about the STATUS of something when only action tools exist, asking to do something to an entity the tools don't cover. "
     "The query should sound natural and plausible — a real user might reasonably expect this to work. answers must be []"),
    ("indirect", "1-2 tool calls — but the query is an INDIRECT or IMPLICIT statement, NOT a direct command. "  # 2/18 ≈ 11%
     "The user describes a situation, feeling, observation, or complaint and the assistant must INFER which tool to call. "
     "Examples: 'it's freezing in here' → set_thermostat higher, 'I can barely see my screen' → set_brightness higher, "
     "'it's so loud' → set_volume lower, 'I'm bored' → play_music or get_news, "
     "'my phone is about to die' → toggle_power_saving on, 'I can't sleep' → toggle_bedtime_mode or start_sleep_tracking, "
     "'I'm lost' → get_directions or share_location, 'this room is stuffy' → control_fan on. "
     "The query must NEVER contain the tool name or action verb directly — it should be a natural human observation or complaint. "
     "Vary between statements ('it's dark'), questions ('why is it so hot?'), and complaints ('ugh this is way too bright')."),
    ("indirect", "1-2 tool calls — but the query is an INDIRECT or IMPLICIT statement, NOT a direct command. "
     "The user describes a situation, feeling, observation, or complaint and the assistant must INFER which tool to call. "
     "Examples: 'it's freezing in here' → set_thermostat higher, 'I can barely see my screen' → set_brightness higher, "
     "'it's so loud' → set_volume lower, 'I'm bored' → play_music or get_news, "
     "'my phone is about to die' → toggle_power_saving on, 'I can't sleep' → toggle_bedtime_mode or start_sleep_tracking, "
     "'I'm lost' → get_directions or share_location, 'this room is stuffy' → control_fan on. "
     "The query must NEVER contain the tool name or action verb directly — it should be a natural human observation or complaint. "
     "Vary between statements ('it's dark'), questions ('why is it so hot?'), and complaints ('ugh this is way too bright')."),
    ("disfluent", "1-2 tool calls — but the query must contain NATURAL SPEECH DISFLUENCIES like real voice transcriptions. "  # 2/20 ≈ 10%
     "Include filler words (um, uh, hmm, like, so, well, okay, right), false starts ('set a — no wait, make that...'), "
     "self-corrections ('send it to — no wait, the other one'), hedging ('I think maybe', 'if possible', 'or whatever'), "
     "trailing off ('can you like... turn off the...'), repetitions ('turn turn off the lights'), "
     "and casual padding ('oh yeah', 'hey so', 'alright so', 'wait actually'). "
     "The underlying intent should still be clear and map to a valid tool call — the disfluencies are just surface noise. "
     "Mix short disfluent queries ('uh lights off') with longer rambly ones."),
    ("disfluent", "1-2 tool calls — but the query must contain NATURAL SPEECH DISFLUENCIES like real voice transcriptions. "
     "Include filler words (um, uh, hmm, like, so, well, okay, right), false starts ('set a — no wait, make that...'), "
     "self-corrections ('send it to — no wait, the other one'), hedging ('I think maybe', 'if possible', 'or whatever'), "
     "trailing off ('can you like... turn off the...'), repetitions ('turn turn off the lights'), "
     "and casual padding ('oh yeah', 'hey so', 'alright so', 'wait actually'). "
     "The underlying intent should still be clear and map to a valid tool call — the disfluencies are just surface noise. "
     "Mix short disfluent queries ('uh lights off') with longer rambly ones."),
    ("garbled", "1-2 tool calls — but the query must contain REALISTIC TYPOS and SPEECH-TO-TEXT ERRORS, as if transcribed by an imperfect ASR system or typed hastily on a phone keyboard. "  # 2/22 ≈ 9%
     "Include: swapped/missing/doubled letters ('trun' for 'turn', 'mesage' for 'message', 'tiemr' for 'timer'), "
     "autocorrect mistakes ('set a duck timer' for 'set a 10 minute timer'), "
     "missing spaces ('turnoff the lights', 'setanalarm'), "
     "homophones from speech ('there' for 'their', 'weather' for 'whether', 'to' for 'two', 'four' for 'for'), "
     "dropped words ('set timer 5' instead of 'set a timer for 5 minutes'), "
     "and phonetic misspellings from voice ('cud you' for 'could you', 'wanna' for 'want to', 'gonna' for 'going to'). "
     "Keep 1-3 errors per query — not every word should be wrong. The intent must still be recoverable. "
     "The tool calls in answers must be CORRECT — garbling is only in the query, not the output."),
    ("garbled", "1-2 tool calls — but the query must contain REALISTIC TYPOS and SPEECH-TO-TEXT ERRORS, as if transcribed by an imperfect ASR system or typed hastily on a phone keyboard. "
     "Include: swapped/missing/doubled letters ('trun' for 'turn', 'mesage' for 'message', 'tiemr' for 'timer'), "
     "autocorrect mistakes ('set a duck timer' for 'set a 10 minute timer'), "
     "missing spaces ('turnoff the lights', 'setanalarm'), "
     "homophones from speech ('there' for 'their', 'weather' for 'whether', 'to' for 'two', 'four' for 'for'), "
     "dropped words ('set timer 5' instead of 'set a timer for 5 minutes'), "
     "and phonetic misspellings from voice ('cud you' for 'could you', 'wanna' for 'want to', 'gonna' for 'going to'). "
     "Keep 1-3 errors per query — not every word should be wrong. The intent must still be recoverable. "
     "The tool calls in answers must be CORRECT — garbling is only in the query, not the output."),
    ("no_tools", "NO tool calls — there are NO tools available at all, answers must be []. "
     "Generate diverse, specific queries as if the user expects tools to exist. Every query must be UNIQUE."),
]

MODEL = "gemini-3.1-flash-lite-preview"

LANGUAGES = [
    "English",
    "Bulgarian", "Croatian", "Czech", "Danish", "Dutch",
    "Estonian", "Finnish", "French", "German", "Greek", "Hungarian",
    "Italian", "Latvian", "Lithuanian", "Maltese", "Polish",
    "Portuguese", "Romanian", "Slovak", "Slovenian", "Spanish",
    "Swedish", "Russian", "Ukrainian",
]

MAX_TOOLS = 10

_TOOL_COUNT_WEIGHTS = {
    1: 8, 2: 10, 3: 14, 4: 14, 5: 14, 6: 12, 7: 10, 8: 8, 9: 5, 10: 5,
}
_TOOL_COUNTS = list(_TOOL_COUNT_WEIGHTS.keys())
_TOOL_WEIGHTS = list(_TOOL_COUNT_WEIGHTS.values())


def _pick_tools(rng, force_empty=False, few_tools=False):
    """Pick tools from 1-3 random pools, with explicit control over total count.

    If few_tools=True, pick only 1-3 tools from a single pool (for multi-call
    scenarios where the same tools are called multiple times with different args).
    If force_empty=True, return no tools.
    Otherwise, sample a target count from _TOOL_COUNT_WEIGHTS and fill from pools.
    """
    if force_empty:
        return []
    if few_tools:
        pool = rng.choice(ALL_POOLS)
        k = rng.randint(1, min(3, len(pool)))
        return rng.sample(pool, k)

    # Pick target tool count from explicit distribution
    target = rng.choices(_TOOL_COUNTS, weights=_TOOL_WEIGHTS, k=1)[0]

    # Draw from pools, adding more pools if needed to meet the target count
    pool_order = list(range(len(ALL_POOLS)))
    rng.shuffle(pool_order)

    candidates = []
    seen = set()
    for pi in pool_order:
        for t in ALL_POOLS[pi]:
            if t["name"] not in seen:
                seen.add(t["name"])
                candidates.append(t)
        if len(candidates) >= target:
            break

    rng.shuffle(candidates)
    return candidates[:target]


# Random context seeds to inject diversity into each batch's prompt.
# Each batch gets one — forces Gemini into a different user/situation space.
_CONTEXT_SEEDS = [
    "The user is a busy parent managing kids and household",
    "The user is a college student studying for finals",
    "The user is commuting on public transit",
    "The user is cooking dinner while multitasking",
    "The user is an elderly person who isn't very tech-savvy",
    "The user is a teenager using their phone casually",
    "The user is a remote worker in a home office",
    "The user is at the gym between sets",
    "The user is traveling abroad and unfamiliar with the area",
    "The user is a professional in back-to-back meetings all day",
    "The user is getting ready for bed and winding down",
    "The user is driving and using voice commands hands-free",
    "The user is a small business owner managing their shop",
    "The user is hosting a dinner party with guests arriving soon",
    "The user is a freelancer juggling multiple client projects",
    "The user is on a hiking trail with limited connectivity",
    "The user is a nurse on a busy hospital shift",
    "The user is a musician setting up for a live performance",
    "The user is a pet owner managing their dog's routine",
    "The user is moving into a new apartment this weekend",
    "The user is planning a surprise birthday party",
    "The user is a photographer on location at a shoot",
    "The user is a teacher preparing for tomorrow's class",
    "The user is recovering from surgery and resting at home",
    "The user is at an airport waiting for a delayed flight",
    "The user is a gamer streaming on Twitch",
    "The user is gardening outside on a sunny afternoon",
    "The user is a new employee on their first week at work",
    "The user is babysitting their niece and nephew",
    "The user is a rideshare driver between pickups",
]


def build_prompt(batch_size, call_desc, tools, rng, query_length_hint=None, language="English"):
    """Build a prompt asking Gemini to generate a batch of examples."""
    scenarios_sample = rng.sample(SCENARIOS, min(20, len(SCENARIOS)))
    scenarios_str = "\n".join(f"  - {s}" for s in scenarios_sample)
    context_seed = rng.choice(_CONTEXT_SEEDS)

    length_instruction = ""
    if query_length_hint:
        length_instruction = f"\n- Query length for this batch: {query_length_hint}"

    if language != "English":
        language_instruction = (
            f"\n- LANGUAGE: ALL queries MUST be written in {language}. "
            f"Use natural, native-sounding {language} — not literal translations from English. "
            f"Include culturally appropriate names, places, idioms, and references for {language}-speaking users. "
            f"Tool names and parameter keys stay in English (they are API identifiers), but string argument VALUES "
            f"(message text, notes, search queries, labels, destinations, etc.) should be in {language} where a real user would use {language}."
        )
    else:
        language_instruction = ""

    if tools:
        tools_section = f"AVAILABLE TOOLS:\n{json.dumps(tools, separators=(',', ':'))}"

        # Check if overlapping tools are present for disambiguation instruction
        tool_names = [t["name"] for t in tools]
        has_overlaps = any(
            a["name"] in tool_names and b["name"] in tool_names
            for a, b in _OVERLAP_PAIRS
        )
        disambig_rule = ""
        if has_overlaps:
            disambig_rule = ("\n- IMPORTANT: Some tools in this set have OVERLAPPING functionality. "
                             "When generating queries, make sure the user's intent clearly maps to ONE specific tool. "
                             "For ambiguous cases where the query could match either tool, pick the MOST specific one "
                             "(e.g. 'text mom' → send_text_message, 'message mom on WhatsApp' → send_instant_message).")

        tool_rules = f"""- Tool call format: {{"name": "tool_name", "arguments": {{"param": "value"}}}}
- Arguments must match the parameter schemas exactly — use correct types (string, number, boolean)
- Do NOT invent tools not in the list — only use the tools shown above
- For contact_id params, use realistic placeholders like "contact_firstname_NNN" with varied names and numbers — never repeat the same contact_id across examples
- For evaluate_js, write real working JavaScript with console.log()
- Number params should be actual numbers not strings (e.g. 7 not "7")
- Boolean params should be actual booleans not strings (e.g. true not "true")
- Vary argument values widely — don't repeat the same locations, names, times, or phrases across examples
- Sometimes include optional parameters, sometimes omit them — mix it up naturally
- Never produce partial tool calls — every call must have "name" and "arguments" with all required params
- Include REALISTIC argument values: names from MANY diverse cultures and languages (Arabic, East Asian, South Asian, European, African, Latin American — never repeat the same name across examples in this batch), emoji in messages, decimal numbers, varied time formats ('7am', '19:30', 'quarter past 3')
- CRITICAL GROUNDING RULE: argument values must come DIRECTLY from the query. Do NOT invent specific names, brands, numbers, or details the user didn't say. If the user says "streaming service", use "streaming service" not "Netflix". If they say "furniture store", use "furniture store" not "IKEA". If they say "set a timer for the pizza", use label "pizza" not "pepperoni pizza". The user's EXACT words should appear in the arguments. Only contact_id params may use placeholders like "contact_alice_123".
- For album names, labels, titles: use ONLY words the user actually said — do not embellish or add creative flourishes{disambig_rule}"""
    else:
        tools_section = "AVAILABLE TOOLS: NONE — no tools are available."
        tool_rules = "- Since no tools are available, ALL answers must be empty arrays []"

    return f"""Generate {batch_size} diverse on-device assistant tool-calling training examples for phones, wearables, and computers.

USER CONTEXT HINT (use as inspiration for ~half the examples, vary the rest freely): {context_seed}

{tools_section}

REQUIREMENTS:
- Each example: a "query" (user's natural language) and "answers" (JSON tool calls array)
- This batch: {call_desc}
- Queries should sound like real users talking to a phone/computer/watch voice assistant or typing a quick command{length_instruction}{language_instruction}
- Vary style: casual, terse, polite ("Could you please..."), conversational, indirect ("it's dark in here" meaning turn on lights)
{tool_rules}
- For empty call examples, answers must be []
- Every query must be UNIQUE — do not repeat patterns or rephrase the same intent
- CRITICAL: use DIFFERENT tools across examples in this batch — do NOT generate multiple examples that all call the same tool. Spread usage across ALL available tools. If there are 6 tools, each should appear in at least one example
- Vary which tools are used and how many are called per example — some examples should use tools from different ends of the list

THEME INSPIRATION — use these as starting points, but generate your OWN unique queries with different names, places, numbers, and details every time. NEVER copy these verbatim:
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



# Overlapping tool pairs — semantically similar tools that require disambiguation
_OVERLAP_PAIRS = [
    (
        {"name": "send_text_message", "description": "Send a text message to a contact via SMS.", "parameters": {"contact_id": {"type": "string", "description": "The contact to message.", "required": True}, "text": {"type": "string", "description": "The message text.", "required": True}}},
        {"name": "send_instant_message", "description": "Send an instant message to a contact.", "parameters": {"contact_id": {"type": "string", "description": "The unique identifier of the recipient contact.", "required": True}, "text": {"type": "string", "description": "The message text to send.", "required": True}}},
    ),
    (
        {"name": "send_email", "description": "Send an email to a recipient.", "parameters": {"to": {"type": "string", "description": "The recipient's email address.", "required": True}, "subject": {"type": "string", "description": "The email subject line.", "required": True}, "body": {"type": "string", "description": "The email body text.", "required": True}}},
        {"name": "compose_email", "description": "Compose and send an email.", "parameters": {"to": {"type": "string", "description": "Recipient email address.", "required": True}, "subject": {"type": "string", "description": "Email subject line.", "required": True}, "body": {"type": "string", "description": "Email body text.", "required": True}, "cc": {"type": "string", "description": "Optional CC recipients.", "required": False}}},
    ),
    (
        {"name": "create_reminder", "description": "Create a reminder for the user at a specific time.", "parameters": {"message": {"type": "string", "description": "The reminder message.", "required": True}, "date_time_human": {"type": "string", "description": "The date and/or time for the reminder.", "required": False}}},
        {"name": "create_list_item", "description": "Add a new item to a user's list with an optional reminder.", "parameters": {"list_name": {"type": "string", "description": "Short name of the list.", "required": True}, "message": {"type": "string", "description": "The text of the list item.", "required": True}, "reminder_date_time_human": {"type": "string", "description": "Optional reminder date/time.", "required": False}}},
    ),
    (
        {"name": "get_directions", "description": "Get directions to a destination.", "parameters": {"destination": {"type": "string", "description": "The destination address or place name.", "required": True}, "mode": {"type": "string", "description": "Travel mode: 'driving', 'walking', 'transit', 'cycling'.", "required": False}}},
        {"name": "start_navigation", "description": "Start turn-by-turn navigation to a destination.", "parameters": {"destination": {"type": "string", "description": "The destination.", "required": True}, "mode": {"type": "string", "description": "Travel mode.", "required": False}, "avoid_tolls": {"type": "boolean", "description": "Whether to avoid toll roads.", "required": False}}},
    ),
    (
        {"name": "play_music", "description": "Play music by song name, artist, album, or genre.", "parameters": {"query": {"type": "string", "description": "Search query: song title, artist name, album, or genre.", "required": True}}},
        {"name": "play_playlist", "description": "Play a specific playlist by name.", "parameters": {"playlist_name": {"type": "string", "description": "Name of the playlist to play.", "required": True}}},
    ),
    (
        {"name": "set_timer", "description": "Set a timer for the specified duration or end time.", "parameters": {"time_human": {"type": "string", "description": "The duration or target end time in human readable format.", "required": True}}},
        {"name": "set_cooking_timer", "description": "Set a named cooking timer.", "parameters": {"label": {"type": "string", "description": "Label for the timer.", "required": True}, "minutes": {"type": "number", "description": "Timer duration in minutes.", "required": True}}},
    ),
]


_QUERY_LENGTH_DESCS = [
    "ultra-short (1-5 words, like 'lights off', 'timer 5', 'call mom')",
    "short (5-10 words, like 'set an alarm for 7am tomorrow')",
    "medium (10-20 words, a normal sentence or two)",
    "long conversational (20+ words, with filler, context, or explanation)",
]


# Words that are common filler / can reasonably appear in argument values
# without being in the query (articles, prepositions, generic action words).
_GROUNDING_STOPWORDS = frozenset({
    "a", "an", "the", "of", "to", "in", "on", "at", "for", "and", "or",
    "is", "it", "my", "me", "i", "you", "your", "this", "that", "with",
    "from", "by", "as", "be", "do", "not", "no", "so", "up", "out",
    "if", "but", "all", "just", "about", "can", "will", "am", "are",
    "was", "were", "been", "has", "have", "had", "would", "could",
    "should", "may", "might", "shall", "its", "our", "we", "they",
    "them", "their", "he", "she", "him", "her", "his", "who", "what",
    "when", "where", "how", "which", "some", "any", "each", "every",
    "also", "than", "then", "now", "here", "there", "very", "too",
    "more", "much", "many", "most", "other", "new", "old", "get",
    "set", "make", "take", "let", "got", "going", "gonna", "want",
    "need", "like", "please", "thanks", "hey", "hi", "ok", "okay",
    "sure", "yes", "no", "true", "false", "null", "none",
    "don't", "doesn't", "didn't", "won't", "can't", "couldn't",
    "shouldn't", "wouldn't", "isn't", "aren't", "wasn't", "weren't",
})

# Param names where values are expected to be generated/inferred, not quoted from query
_FREEFORM_PARAMS = frozenset({
    "contact_id", "password", "js", "url", "source", "confirmation_code",
    "order_id", "post_id",
})


def _grounding_check(pname, pval, pdesc, query, query_lower):
    """Check that significant words in a string argument value are grounded in the query.

    Returns False if the value contains specific details not traceable to the query.
    Strict: every specific noun/name/identifier must come from the query.
    """
    # Skip params where values are generated/inferred, not extracted from query
    if pname in _FREEFORM_PARAMS:
        return True

    val = pval.strip()
    if not val:
        return True

    # Skip if the param description suggests free-form/generated content
    pdesc_lower = pdesc.lower() if pdesc else ""
    freeform_hints = ("url", "link", "password", "code", "javascript", "js ",
                      "email address", "phone number")
    if any(hint in pdesc_lower for hint in freeform_hints):
        return True

    # For enum-like values that match description options, always allow
    # (e.g., "driving", "bake", "front" — these are valid fixed choices)
    quoted = re.findall(r"'([^']+)'", pdesc_lower)
    if quoted and val.lower() in [q.lower() for q in quoted]:
        return True

    # Extract significant words from value and query
    # Covers Latin, Latin Extended, Cyrillic, Greek, and combining diacriticals
    _WORD_RE = r'[\w\u00C0-\u024F\u0370-\u03FF\u0400-\u04FF\u0500-\u052F]{3,}'
    val_words = set()
    for word in re.findall(_WORD_RE, val.lower()):
        if word not in _GROUNDING_STOPWORDS:
            val_words.add(word)

    if not val_words:
        return True

    query_words = set()
    for word in re.findall(_WORD_RE, query_lower):
        if word not in _GROUNDING_STOPWORDS:
            query_words.add(word)

    # Check grounding: every significant value word must trace to the query
    grounded = 0
    for vw in val_words:
        if vw in query_words:
            grounded += 1
            continue
        # Substring/stem match (catches "grocery"/"groceries", "alarm"/"alarms")
        if any(vw in qw or qw in vw for qw in query_words):
            grounded += 1
            continue
        # Raw substring in query (catches compound words, multi-word names)
        if vw in query_lower:
            grounded += 1
            continue

    # Strict threshold: 75% of value words must be grounded.
    # For short values (≤3 significant words), require ALL to be grounded —
    # this catches "Netflix" from "streaming service", "IKEA" from "furniture store".
    if len(val_words) <= 3:
        if grounded < len(val_words):
            return False
    else:
        grounding_ratio = grounded / len(val_words)
        if grounding_ratio < 0.75:
            return False

    # Additional hard checks for specific fabrication patterns
    # Reject fabricated phone numbers in non-phone params
    phones = re.findall(r'[\+]?[\d\s\-\(\)]{7,}', val)
    if phones and pname not in ("phone", "phone_number"):
        query_digits = set(re.findall(r'\d{3,}', query))
        val_digits = set(re.findall(r'\d{3,}', val))
        if val_digits and not val_digits & query_digits:
            return False

    # Reject fabricated email addresses in non-email params
    emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', val)
    if emails and pname not in ("to", "email", "cc", "bcc", "from_contact"):
        for email in emails:
            local = email.split("@")[0].replace(".", " ").replace("_", " ")
            if not any(w.lower() in query_lower for w in local.split() if len(w) > 2):
                return False

    return True


def _semantic_check(tool_name, args, schema, query, call_type="single"):
    """Lightweight rule-based semantic validation of argument values.

    Returns False if any argument value is obviously nonsensical.
    Catches the most common Gemini hallucination patterns without needing an LLM judge.
    """
    query_lower = query.lower()

    for pname, pval in args.items():
        pinfo = schema.get(pname, {})
        expected_type = pinfo.get("type", "string")
        pdesc = pinfo.get("description", "").lower()

        # Number range checks 
        if expected_type == "number" and isinstance(pval, (int, float)):
            if any(kw in pname for kw in ("level", "brightness", "percent")):
                if pval < 0 or pval > 100:
                    return False
            if pname == "temperature" and "thermostat" in tool_name:
                if pval < 40 or pval > 100: 
                    return False
            if any(kw in pname for kw in ("minutes", "duration", "hours", "seconds", "count")):
                if pval < 0:
                    return False
            if pname == "rating":
                if pval < 1 or pval > 5:
                    return False

        # String emptiness checks
        if expected_type == "string" and isinstance(pval, str):
            if pinfo.get("required") and not pval.strip():
                return False
            val_lower = pval.strip().lower()
            if val_lower in ("", "null", "none", "undefined", "n/a", "todo",
                             "example", "test", "placeholder", "your_",
                             "insert", "[", "{", "..."):
                if pinfo.get("required"):
                    return False

        # Enum-like string checks from description 
        if expected_type == "string" and isinstance(pval, str) and pdesc:
            quoted = re.findall(r"'([^']+)'", pdesc)
            if len(quoted) >= 2:  # looks like an enum
                if pval.lower() not in [q.lower() for q in quoted]:
                    if len(pval) < 30:
                        return False

        # check alignment with query intent
        if expected_type == "boolean" and isinstance(pval, bool):
            if pname == "enabled":
                disable_words = ("off", "disable", "stop", "don't", "no ", "without")
                enable_words = ("on", "enable", "start", "turn on", "activate")
                query_wants_off = any(w in query_lower for w in disable_words)
                query_wants_on = any(w in query_lower for w in enable_words)
                # Only reject clear contradictions — if ambiguous, allow
                if query_wants_off and not query_wants_on and pval is True:
                    return False
                if query_wants_on and not query_wants_off and pval is False:
                    return False

        # Grounding check: argument values must be traceable to the query.
        # Skip for call types where semantic inference is the point (indirect,
        # disfluent, garbled) — those intentionally have values not literally in the query.
        # For multi_long_values, only skip grounding on free-text content params
        # (message bodies, notes, etc.) — not on identifier params like names/sites.
        _SKIP_GROUNDING_TYPES = ("indirect", "disfluent", "garbled")
        _LONG_CONTENT_PARAMS = frozenset({
            "text", "body", "message", "note", "new_text", "items",
            "description", "feedback", "actions", "js",
        })
        skip_grounding = (
            call_type in _SKIP_GROUNDING_TYPES
            or (call_type == "multi_long_values" and pname in _LONG_CONTENT_PARAMS)
        )
        if expected_type == "string" and isinstance(pval, str) and not skip_grounding:
            ok = _grounding_check(pname, pval, pdesc, query, query_lower)
            if not ok:
                return False

    return True


# Description rephrase templates — vary how tool descriptions are worded
# without changing the meaning. Prevents the model from memorizing exact phrasing.
_DESC_TEMPLATES = [
    lambda d: d,  # original
    lambda d: d,  # original (weighted higher)
    lambda d: d,  # original (weighted higher)
    lambda d: d.rstrip(".") + "." if not d.endswith(".") else d,  # normalize period
    lambda d: d[0].upper() + d[1:] if d and d[0].islower() else d,  # capitalize
    lambda d: d.replace(".", ",", 1).rstrip(",") + "." if d.count(".") > 1 else d,  # comma variant
    lambda d: d.split(".")[0].strip() + "." if "." in d else d,  # first sentence only
    lambda d: d.replace(" the ", " a ", 1) if " the " in d else d,  # article swap
    lambda d: d.replace("Enable or disable", "Toggle").replace("enable or disable", "toggle") if "nable or disable" in d else d,
    lambda d: d.replace("Turn ", "Switch ").replace("turn ", "switch ") if d.startswith(("Turn", "turn")) else d,
]


# Domain categories for tool synthesis — Gemini invents tools within these bounds
_SYNTH_DOMAINS = [
    "smart kitchen appliances (oven, fridge, blender, air purifier, water filter)",
    "pet care and pet tech (feeder, tracker, health monitor, activity log)",
    "personal finance and budgeting (subscriptions, savings goals, expense tracking)",
    "study and education tools (flashcards, quiz, study timer, citation generator)",
    "creative tools (drawing, music composition, video editing, photo collage)",
    "gardening and plant care (watering schedule, soil sensor, plant identifier, weather alerts)",
    "meeting and collaboration (screen share, whiteboard, poll, action items, transcription)",
    "home maintenance (HVAC filter, smoke detector, water leak sensor, appliance warranty)",
    "sports and outdoor activities (route planner, gear checklist, weather for trail, pace tracker)",
    "baby and childcare (feeding log, diaper tracker, sleep monitor, milestone tracker)",
    "subscription and account management (cancel subscription, change plan, check renewal date)",
    "custom device peripherals (drone control, 3D printer, smart mirror, e-ink display)",
    "language learning (vocabulary drill, pronunciation check, daily lesson, streak tracker)",
    "car maintenance (oil change reminder, tire rotation schedule, mileage log, service history)",
    "event planning (guest list, RSVP tracker, venue search, seating chart, send invitations)",
]

_SYNTH_TOOL_PROMPT = """Generate {num_tools} realistic tool definitions for a consumer device assistant in this domain: {domain}

Each tool must have:
- "name": snake_case function name
- "description": clear 1-sentence description of what it does
- "parameters": dict of param_name -> {{"type": "string"|"number"|"boolean", "description": "...", "required": true|false}}

Rules:
- Tools should feel like real device/app APIs — specific, actionable, not vague
- Mix of tools with 0-4 parameters
- Include both action tools (do something) and query tools (get information)
- Parameter types must be realistic (numbers for quantities, booleans for toggles, strings for names/text)
- Use snake_case for all names

Return ONLY a valid JSON array of tool objects. No markdown, no explanation."""


_synth_stats = {"attempted": 0, "succeeded": 0, "failed_parse": 0, "failed_validate": 0, "failed_exception": 0}


def _synthesize_tools(client_pool, rng, model, num_tools):
    """Ask Gemini to invent novel tool definitions within a random domain."""
    _synth_stats["attempted"] += 1
    domain = rng.choice(_SYNTH_DOMAINS)
    prompt = _SYNTH_TOOL_PROMPT.format(num_tools=num_tools, domain=domain)

    client = client_pool.get()
    try:
        response = client.models.generate_content(
            model=model, contents=prompt,
            config={"temperature": 0.9, "max_output_tokens": 4096},
        )
        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        tools = json.loads(text.strip())
        if not isinstance(tools, list):
            _synth_stats["failed_parse"] += 1
            return None
        # Validate and normalize structure to match our pool format
        valid = []
        for t in tools:
            if not isinstance(t, dict):
                continue
            if not t.get("name") or not isinstance(t.get("name"), str):
                continue
            if not t.get("description") or not isinstance(t.get("description"), str):
                continue
            # Ensure snake_case name
            t["name"] = re.sub(r'[^a-z0-9_]', '_', t["name"].lower()).strip('_')
            # Normalize parameters to match our schema format
            raw_params = t.get("parameters", {})
            if not isinstance(raw_params, dict):
                t["parameters"] = {}
            else:
                normalized = {}
                for pname, pinfo in raw_params.items():
                    if isinstance(pinfo, dict):
                        normalized[pname] = {
                            "type": pinfo.get("type", "string"),
                            "description": pinfo.get("description", ""),
                            "required": pinfo.get("required", False),
                        }
                    elif isinstance(pinfo, str):
                        # Gemini sometimes produces {"param": "description string"}
                        normalized[pname] = {
                            "type": "string",
                            "description": pinfo,
                            "required": True,
                        }
                t["parameters"] = normalized
            valid.append(t)
        if not valid:
            _synth_stats["failed_validate"] += 1
            return None
        _synth_stats["succeeded"] += 1
        return valid[:num_tools]
    except Exception as e:
        _synth_stats["failed_exception"] += 1
        return None


def _rephrase_tool_descriptions(tools, rng):
    """Apply random description rephrasing to tool definitions."""
    result = []
    for t in tools:
        t = dict(t)  # shallow copy
        template = rng.choice(_DESC_TEMPLATES)
        t["description"] = template(t["description"])
        # Also occasionally rephrase parameter descriptions
        if isinstance(t.get("parameters"), dict):
            params = {}
            for pname, pinfo in t["parameters"].items():
                pinfo = dict(pinfo)
                if rng.random() < 0.3 and "description" in pinfo:
                    pinfo["description"] = rng.choice(_DESC_TEMPLATES)(pinfo["description"])
                params[pname] = pinfo
            t["parameters"] = params
        result.append(t)
    return result


def generate_batch(client_pool, batch_size, rng, model, language="English"):
    """Generate one batch of examples. Returns list of dicts."""
    call_type, call_desc = rng.choice(CALL_TYPES)

    # ~15% of batches use synthesized novel tools for generalization
    # Only for call types that benefit from multiple tools (not few_tools types)
    use_synth = (
        call_type not in ("no_tools", "multi_few_tools")
        and rng.random() < 0.50
    )

    if use_synth:
        # Synthesize at least 3 tools so we get variety, up to 8
        target = max(3, rng.choices(_TOOL_COUNTS, weights=_TOOL_WEIGHTS, k=1)[0])
        target = min(target, 8)
        tools = _synthesize_tools(client_pool, rng, model, target)
        if tools is None:
            use_synth = False
            tools = _pick_tools(rng, force_empty=False,
                                few_tools=(call_type in ("multi_few_tools", "multi_long_values")))
    else:
        tools = _pick_tools(
            rng,
            force_empty=(call_type == "no_tools"),
            few_tools=(call_type in ("multi_few_tools", "multi_long_values")),
        )

    # Inject overlapping tools ~20% of the time (only for pool tools)
    if tools and not use_synth and len(tools) >= 2 and rng.random() < 0.20:
        pair = rng.choice(_OVERLAP_PAIRS)
        idx = rng.randint(0, len(tools) - 1)
        tools[idx] = pair[0]
        if len(tools) < MAX_TOOLS:
            tools.insert(idx + 1, pair[1])
        else:
            other = (idx + 1) % len(tools)
            tools[other] = pair[1]

    # Rephrase tool descriptions ~30% of the time for wording diversity
    if tools and rng.random() < 0.30:
        tools = _rephrase_tool_descriptions(tools, rng)

    # Pick a query length bucket for this batch
    query_length_hint = rng.choice(_QUERY_LENGTH_DESCS)

    prompt = build_prompt(batch_size, call_desc, tools, rng, query_length_hint=query_length_hint, language=language)

    # Vary temperature per batch for diversity: mostly 0.9-1.1 with occasional
    # low (more precise args) and high (more creative queries) extremes.
    temperature = rng.choice([0.7, 0.8, 0.9, 1.0, 1.0, 1.0, 1.1, 1.1, 1.2, 1.3])

    client = client_pool.get()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config={
            "temperature": temperature,
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
    tools_str = json.dumps(tools, separators=(",", ":"), ensure_ascii=False)
    tool_name_set = {t["name"] for t in tools}

    # Build schema map for parameter validation: {tool_name: {param_name: {type, required}}}
    tool_schema = {}
    for t in tools:
        params = t.get("parameters", {})
        tool_schema[t["name"]] = {
            k: {"type": v.get("type", "string"), "required": v.get("required", False)}
            for k, v in params.items()
        } if isinstance(params, dict) else {}

    # Build keyword set for rejecting bad empty-answer examples
    # Maps tool descriptions to simple action words for fuzzy matching
    _tool_keywords = set()
    for t in tools:
        _tool_keywords.add(t["name"].replace("_", " "))
        for word in t.get("description", "").lower().split():
            if len(word) > 4 and word not in ("the", "from", "with", "that", "this", "about", "which", "their", "optional"):
                _tool_keywords.add(word)

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
            cname = call.get("name")
            if cname not in tool_name_set:
                ok = False
                break
            args = call.get("arguments", {})
            if not isinstance(args, dict):
                ok = False
                break

            # Validate required params are present
            schema = tool_schema.get(cname, {})
            for pname, pinfo in schema.items():
                if pinfo["required"] and pname not in args:
                    ok = False
                    break

            # Validate and coerce argument types
            if ok:
                for pname, pval in list(args.items()):
                    if pname not in schema:
                        continue  # extra params — tolerate
                    expected = schema[pname]["type"]
                    if expected == "number" and isinstance(pval, str):
                        try:
                            args[pname] = float(pval) if "." in pval else int(pval)
                        except ValueError:
                            ok = False
                            break
                    elif expected == "boolean" and isinstance(pval, str):
                        if pval.lower() in ("true", "false"):
                            args[pname] = pval.lower() == "true"
                        else:
                            ok = False
                            break

            # Lightweight semantic checks on argument values
            if ok:
                ok = _semantic_check(cname, args, schema, query, call_type)

            if not ok:
                break
        if not ok:
            continue

        # Reject empty answers that look like they should have tool calls.
        # Skip this filter for "near_miss" — those intentionally have related keywords.
        if len(answers) == 0 and tools and call_type == "none":
            query_lower = query.lower()
            if any(kw in query_lower for kw in _tool_keywords if len(kw) > 5):
                continue  # skip — query matches a tool but answer is empty

        valid.append({
            "query": query,
            "tools": tools_str,
            "answers": json.dumps(answers, separators=(",", ":"), ensure_ascii=False),
            "source": "synth-gemini-assistant",
            "model": model,
            "call_type": call_type,
            "num_tools": len(tools),
            "synth_tools": use_synth,
            "language": language,
        })

    return valid


def generate_all(num_samples, workers=8, batch_size=25, model=MODEL, client_pool=None):
    """Generate num_samples examples using parallel Gemini calls."""
    if client_pool is None:
        client_pool = ClientPool(make_clients())
    rng = random.Random(42)

    target = int(num_samples * 1.3)
    max_submissions = target // batch_size * 3  # hard cap to prevent infinite loop
    all_examples = []
    failed = 0

    pbar = tqdm(total=num_samples, desc="Generating", unit="ex")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = set()
        submitted = 0

        def _submit_one():
            nonlocal submitted
            batch_rng = random.Random(rng.randint(0, 2**32))
            language = LANGUAGES[submitted % len(LANGUAGES)]
            f = pool.submit(generate_batch, client_pool, batch_size, batch_rng, model, language=language)
            pending.add(f)
            submitted += 1

        initial = min(workers, (target + batch_size - 1) // batch_size)
        for _ in range(initial):
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
                pbar.n = min(len(all_examples), num_samples)
                pbar.set_postfix(failed=failed)
                pbar.refresh()
                if len(all_examples) < target and submitted < max_submissions:
                    _submit_one()

    pbar.close()

    def _dedup_key(query):
        """Normalize query for dedup: lowercase, strip punctuation/whitespace."""
        return " ".join(query.lower().split()).strip(".,!?;:'\"")

    seen = set()
    deduped = []
    for ex in all_examples:
        key = _dedup_key(ex["query"])
        if key not in seen:
            seen.add(key)
            deduped.append(ex)

    if len(deduped) > num_samples:
        deduped = deduped[:num_samples]

    # Log distribution stats
    from collections import Counter
    type_counts = Counter(ex.get("call_type", "unknown") for ex in deduped)
    tool_counts = Counter(ex.get("num_tools", -1) for ex in deduped)
    print(f"Generated {len(deduped):,} unique examples ({len(all_examples) - len(deduped)} duplicates removed, {failed} failed batches)")
    print(f"  Call type distribution: {dict(type_counts.most_common())}")
    print(f"  Tool count distribution: {dict(sorted(tool_counts.items()))}")
    lang_counts = Counter(ex.get("language", "English") for ex in deduped)
    print(f"  Language distribution: {dict(lang_counts.most_common())}")
    if _synth_stats["attempted"] > 0:
        print(f"  Tool synthesis: {_synth_stats['succeeded']}/{_synth_stats['attempted']} succeeded "
              f"(parse_fail={_synth_stats['failed_parse']}, validate_fail={_synth_stats['failed_validate']}, "
              f"exception={_synth_stats['failed_exception']})")
    return deduped



LOCAL_UNIFIED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "tool_calls_unified")
HF_DATASET_REPO = "Cactus-Compute/tool-calls"
UPLOAD_EVERY = 100000


def _load_existing():
    """Load existing dataset from disk or HuggingFace."""
    from datasets import load_from_disk

    local = os.path.abspath(LOCAL_UNIFIED_DIR)
    if os.path.exists(local) and any(f.endswith(".arrow") for f in os.listdir(local)):
        ds = load_from_disk(local)
        print(f"Loaded existing dataset: {len(ds)} rows")
    else:
        print(f"Downloading existing dataset from {HF_DATASET_REPO}...")
        from .dataset import download_hf_split
        try:
            ds = download_hf_split("train", HF_DATASET_REPO)
        except Exception as exc:
            print(
                f"Warning: could not load existing dataset {HF_DATASET_REPO}: {exc}\n"
                "Generation will continue from an empty corpus and save results locally."
            )
            return None
        # Save locally so subsequent chunks don't re-download
        os.makedirs(local, exist_ok=True)
        ds.save_to_disk(local)
        # Clear HF download cache to avoid double storage
        hf_cache = os.path.expanduser("~/.cache/huggingface/datasets")
        if os.path.exists(hf_cache):
            import shutil
            shutil.rmtree(hf_cache)
            print(f"Cleared HF dataset cache ({hf_cache})")
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
        "language": [ex.get("language", "English") for ex in new_examples],
    })
    if existing is None:
        merged = new_ds
        print(f"\nCreated dataset: {len(merged)} rows")
    else:
        # Keep only columns that exist in the existing dataset (language may be new)
        extra_cols = [c for c in new_ds.column_names if c not in existing.column_names]
        if extra_cols:
            # Add missing columns to existing dataset with defaults
            for col in extra_cols:
                existing = existing.add_column(col, ["English"] * len(existing) if col == "language" else [""] * len(existing))
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

    if existing is None:
        print(f"Skipping upload to {HF_DATASET_REPO}; no existing remote dataset was available.")
        return merged

    import tempfile
    from huggingface_hub import HfApi, CommitOperationDelete

    api = HfApi()
    api.create_repo(HF_DATASET_REPO, repo_type="dataset", private=False, exist_ok=True)

    # Export to parquet and upload via HfApi to avoid push_to_hub dataset card bug
    parquet_dir = tempfile.mkdtemp(prefix="needle_upload_")
    print(f"Exporting to parquet...")
    merged.to_parquet(os.path.join(parquet_dir, "train.parquet"))

    # Delete old train shards first
    files = api.list_repo_files(HF_DATASET_REPO, repo_type="dataset", token=True)
    old_train = [f for f in files if f.startswith("data/train-")]
    if old_train:
        print(f"Deleting {len(old_train)} old train shards...")
        ops = [CommitOperationDelete(path_in_repo=f) for f in old_train]
        api.create_commit(
            repo_id=HF_DATASET_REPO, repo_type="dataset", operations=ops,
            commit_message="Remove old train shards before re-upload", token=True,
        )

    print(f"Uploading to {HF_DATASET_REPO} (train split)...")
    api.upload_file(
        path_or_fileobj=os.path.join(parquet_dir, "train.parquet"),
        path_in_repo="data/train-00000-of-00001.parquet",
        repo_id=HF_DATASET_REPO, repo_type="dataset", token=True,
        commit_message=f"Upload train data ({len(merged)} rows)",
    )

    import shutil as _shutil
    _shutil.rmtree(parquet_dir)
    print(f"Upload complete: {HF_DATASET_REPO}")

    return merged


def main(args):

    client_pool = ClientPool(make_clients())

    remaining = args.num_samples
    total_generated = 0
    existing = None if args.dry_run else _load_existing()
    def _dedup_key(query):
        return " ".join(query.lower().split()).strip(".,!?;:'\"")

    seen_queries = set()
    if existing and not args.dry_run:
        seen_queries = set(_dedup_key(q) for q in existing["query"])

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
            key = _dedup_key(ex["query"])
            if key not in seen_queries:
                seen_queries.add(key)
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
            # Show synthesized tool examples first (if any)
            synth_examples = [ex for ex in examples if ex.get("synth_tools")]
            if synth_examples:
                print(f"\nSynthesized tool examples ({len(synth_examples)} total):")
                for ex in synth_examples[:10]:
                    try:
                        tool_names = [t["name"] for t in json.loads(ex["tools"])]
                        tools_str = ", ".join(tool_names) if tool_names else "(no tools)"
                    except (json.JSONDecodeError, TypeError):
                        tools_str = "(no tools)"
                    print(f"  Tools: [{tools_str}] [SYNTH]")
                    print(f"  Q: {ex['query']}")
                    print(f"  A: {ex['answers'][:200]}")
                    print()

            print(f"\nSample examples:")
            sample_indices = list(range(len(examples)))
            random.shuffle(sample_indices)
            for idx in sample_indices[:100]:
                ex = examples[idx]
                try:
                    tool_names = [t["name"] for t in json.loads(ex["tools"])]
                    tools_str = ", ".join(tool_names) if tool_names else "(no tools)"
                except (json.JSONDecodeError, TypeError):
                    tools_str = "(no tools)"
                synth_tag = " [SYNTH]" if ex.get("synth_tools") else ""
                print(f"  Tools: [{tools_str}]{synth_tag}")
                print(f"  Q: {ex['query']}")
                print(f"  A: {ex['answers'][:200]}")
                print()

        if args.dry_run:
            print(f"  Chunk: {len(examples)} examples (dry-run, not uploading)")
            continue

        existing = _merge_and_upload(existing, examples)

    print(f"\nDone. Total generated: {total_generated:,}")
    if args.output_jsonl:
        print(f"Raw JSONL: {args.output_jsonl}")


if __name__ == "__main__":
    import argparse as _ap
    _p = _ap.ArgumentParser(description="Generate on-device assistant tool-calling data with Gemini")
    _p.add_argument("--num-samples", type=int, default=500)
    _p.add_argument("--batch-size", type=int, default=10)
    _p.add_argument("--workers", type=int, default=8)
    _p.add_argument("--model", type=str, default=MODEL)
    _p.add_argument("--dry-run", action="store_true")
    _p.add_argument("--output-jsonl", type=str, default=None)
    _p.add_argument("--upload-every", type=int, default=UPLOAD_EVERY)
    main(_p.parse_args())
