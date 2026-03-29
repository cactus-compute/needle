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
    # Ride-hailing & delivery
    "ordering an Uber to the airport", "tracking a food delivery",
    "canceling a ride", "ordering pizza from a nearby place",
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
    # Indirect / implicit intent
    "it's really dark in here", "I can't hear anything on this call",
    "I'm freezing", "it's too bright", "I'm bored",
    "I need to wake up early tomorrow", "I keep forgetting to drink water",
    # Corrections and follow-ups
    "no wait, make that 7:30 instead", "actually send it to Sarah not John",
    "cancel what I just set", "change the timer to 10 minutes",
    "undo that", "never mind, turn it back on",
    # Temporal and conditional
    "every weekday at 8am", "when I get home remind me to call mom",
    "in 2 hours turn off the lights", "tomorrow morning check the weather",
    "after work start the robot vacuum", "at sunset close the blinds",
    # Multi-person and group
    "message both Sarah and John about dinner", "set up a group call with the team",
    "share my location with everyone in the family group",
    "send the same email to Alice, Bob, and Carol",
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
    "send a message to Alice about the meeting AND a message to Bob about the deadline",
    "set an alarm for 6:30am for gym AND another for 8am for work",
    "add milk, bread, and eggs to the shopping list as separate items",
    "create reminders for dentist on Monday, haircut on Wednesday, and vet on Friday",
    "text Sarah I'm running late, text John I'll be there in 20, text Mom dinner is at 7",
    "set timers for pasta 12 minutes, sauce 8 minutes, and garlic bread 5 minutes",
    "send an email to the team about the Q3 report AND another to the client about the proposal deadline",
    "add three different notes: one about the meeting summary, one about the project timeline, and one about the budget",
    "create calendar events for Monday standup at 9, Wednesday review at 2, and Friday retro at 4",
    "log breakfast as oatmeal with blueberries, lunch as grilled chicken salad, dinner as pasta carbonara",
    # Multi-call with long argument values
    "send a detailed message to the team explaining that the deployment is postponed to next Tuesday due to the security review",
    "create a note with the full recipe for chocolate chip cookies including all ingredients and steps",
    "email the landlord about the broken dishwasher explaining what happened and when it started",
    "set a reminder with the full address: 1234 Oak Street, Apartment 5B, Springfield, IL 62704",
    "post a status update about how excited I am for the conference next week and all the sessions I'm looking forward to",
    # Edge cases
    "very short terse command like 'timer 5 min'",
    "long polite request with extra context",
    "ambiguous request that maps to one tool",
    "casual voice assistant style like 'hey set my alarm for seven'",
    "request where no available tool can help",
]

CALL_TYPES = [
    ("single", "exactly 1 tool call"),                                                                      # 3/14 ≈ 21%
    ("single", "exactly 1 tool call"),
    ("single", "exactly 1 tool call"),
    ("multi", "2-3 tool calls (the user wants multiple things done at once)"),                               # 3/14 ≈ 21%
    ("multi", "2-3 tool calls (the user wants multiple things done at once)"),
    ("multi", "2-3 tool calls (the user wants multiple things done at once)"),
    ("multi_few_tools", "2-4 tool calls using ONLY 1-3 of the available tools — the user wants multiple actions with the SAME or very few tools, with DIFFERENT detailed argument values each time (e.g. sending messages to 3 different people, setting 2 different alarms, creating multiple list items). Each call MUST have distinct, realistic argument values — vary names, times, locations, messages, etc."),  # 4/14 ≈ 29%
    ("multi_few_tools", "2-4 tool calls using ONLY 1-3 of the available tools — the user wants multiple actions with the SAME or very few tools, with DIFFERENT detailed argument values each time (e.g. sending messages to 3 different people, setting 2 different alarms, creating multiple list items). Each call MUST have distinct, realistic argument values — vary names, times, locations, messages, etc."),
    ("multi_few_tools", "2-4 tool calls using ONLY 1-3 of the available tools — the user wants multiple actions with the SAME or very few tools, with DIFFERENT detailed argument values each time (e.g. sending messages to 3 different people, setting 2 different alarms, creating multiple list items). Each call MUST have distinct, realistic argument values — vary names, times, locations, messages, etc."),
    ("multi_few_tools", "2-4 tool calls using ONLY 1-3 of the available tools — the user wants multiple actions with the SAME or very few tools, with DIFFERENT detailed argument values each time (e.g. sending messages to 3 different people, setting 2 different alarms, creating multiple list items). Each call MUST have distinct, realistic argument values — vary names, times, locations, messages, etc."),
    ("multi_long_values", "2-3 tool calls where argument values are LONG and detailed — full sentences for message text, multi-word descriptions, specific addresses, complete email bodies, detailed notes. Each argument value should be at least 5-10 words, not just a single word or number."),  # 2/14 ≈ 14%
    ("multi_long_values", "2-3 tool calls where argument values are LONG and detailed — full sentences for message text, multi-word descriptions, specific addresses, complete email bodies, detailed notes. Each argument value should be at least 5-10 words, not just a single word or number."),
    ("none", "NO tool calls — the user asks a question or makes a request that NONE of the available tools can fulfill (e.g. asking for opinions, general knowledge, emotional support, or tasks outside the tool capabilities). The query must NOT be something any of the listed tools could handle. answers must be []"),  # 1/14 ≈ 7%
    ("no_tools", "NO tool calls — there are NO tools available at all, answers must be []"),                 # 1/14 ≈ 7%
]

MODEL = "gemini-3.1-flash-lite-preview"

MAX_TOOLS = 10


def _pick_tools(rng, force_empty=False, few_tools=False):
    """Pick 0-10 tools from 1-3 random pools.

    If few_tools=True, pick only 1-3 tools from a single pool (for multi-call
    scenarios where the same tools are called multiple times with different args).
    """
    if force_empty:
        return []
    if few_tools:
        pool = rng.choice(ALL_POOLS)
        k = rng.randint(1, min(3, len(pool)))
        return rng.sample(pool, k)
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
    scenarios_sample = rng.sample(SCENARIOS, min(20, len(SCENARIOS)))
    scenarios_str = "\n".join(f"  - {s}" for s in scenarios_sample)

    if tools:
        tools_section = f"AVAILABLE TOOLS:\n{json.dumps(tools, separators=(',', ':'))}"
        tool_rules = """- Tool call format: {{"name": "tool_name", "arguments": {{"param": "value"}}}}
- Arguments must match the parameter schemas exactly — use correct types (string, number, boolean)
- Do NOT invent tools not in the list — only use the tools shown above
- For contact_id params, use realistic placeholders like "contact_alice_123", "contact_john_456"
- For evaluate_js, write real working JavaScript with console.log()
- Number params should be actual numbers not strings (e.g. 7 not "7")
- Boolean params should be actual booleans not strings (e.g. true not "true")
- Vary argument values widely — don't repeat the same locations, names, times, or phrases across examples
- Sometimes include optional parameters, sometimes omit them — mix it up naturally
- Never produce partial tool calls — every call must have "name" and "arguments" with all required params"""
    else:
        tools_section = "AVAILABLE TOOLS: NONE — no tools are available."
        tool_rules = "- Since no tools are available, ALL answers must be empty arrays []"

    return f"""Generate {batch_size} diverse on-device assistant tool-calling training examples for phones, wearables, and computers.

{tools_section}

REQUIREMENTS:
- Each example: a "query" (user's natural language) and "answers" (JSON tool calls array)
- This batch: {call_desc}
- Queries should sound like real users talking to a phone/computer/watch voice assistant or typing a quick command
- Vary query length: mix ultra-short ("timer 5 min", "lights off"), medium ("set an alarm for 7am tomorrow"), and long conversational ("hey can you set a timer for like 20 minutes, I'm making pasta")
- Vary style: casual, terse, polite ("Could you please..."), conversational, indirect ("it's dark in here" meaning turn on lights)
{tool_rules}
- For empty call examples, answers must be []
- Every query must be UNIQUE — do not repeat patterns or rephrase the same intent

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
    tools = _pick_tools(
        rng,
        force_empty=(call_type == "no_tools"),
        few_tools=(call_type in ("multi_few_tools", "multi_long_values")),
    )

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
            if call.get("name") not in tool_name_set:
                ok = False
                break
            if not isinstance(call.get("arguments", {}), dict):
                ok = False
                break
        if not ok:
            continue

        # Reject empty answers that look like they should have tool calls
        if len(answers) == 0 and tools and call_type == "none":
            query_lower = query.lower()
            if any(kw in query_lower for kw in _tool_keywords if len(kw) > 5):
                continue  # skip — query matches a tool but answer is empty

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
UPLOAD_EVERY = 10000


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
    print(f"Uploading to {HF_DATASET_REPO} (train split)...")
    merged.push_to_hub(HF_DATASET_REPO, split="train", token=True)
    print(f"Upload complete: {HF_DATASET_REPO}")
    print("NOTE: Run 'python scripts/split_dataset.py' to create the validation split.")

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
        seen_queries = set(existing["query"])

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
            sample_indices = list(range(len(examples)))
            random.shuffle(sample_indices)
            for idx in sample_indices[:20]:
                ex = examples[idx]
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
    main()
