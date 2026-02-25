# Editing Mode Quick Start Guide

## Getting Started

### Prerequisites

- Application running with camera access
- Hands visible to camera

### Basic Workflow

#### Step 1: Capture Screenshot

1. **Form Rectangle:** Use both hands to form a rectangle around the object you want to capture
   - Hold hand landmarks to create 4 corners
   - Rectangle will show as cyan outline on camera feed
2. **Trigger Capture:**
   - **Snap:** Single hand with pinch gesture (thumb + index)
   - **Double-Pinch:** Both hands with pinch gesture within 0.35s window
3. **Automatic Entry:** UI automatically switches to editing mode

#### Step 2: Edit Image

### Brightness Control

- **Slider Range:** -100 (darker) to +100 (brighter)
- **Default:** 0 (no change)
- **Action:** Move slider left/right to darken/brighten
- **Preview:** Updates in real-time

### Contrast Control

- **Slider Range:** 0 (no contrast) to 300 (very high)
- **Default:** 100 (normal contrast)
- **Action:** Move slider to increase/decrease contrast
- **Preview:** Updates in real-time

### Blur Filter

- **Slider Range:** 0 to 50 (strength)
- **Default:** 0 (no blur)
- **Action:** Move slider to blur image
- **Use Case:** Smooth out noise or details

### Sharpen Filter

- **Slider Range:** 0 to 50 (strength)
- **Default:** 0 (no sharpen)
- **Action:** Move slider to sharpen image
- **Use Case:** Enhance details and edges

### Special Filters

Click any filter button to apply instantly:

- **Grayscale:** Convert to black & white
- **Sepia:** Apply warm vintage tone
- **HSV Eq:** Histogram equalization for better contrast

### Undo/Redo

- **Undo (↶):** Revert last operation
- **Redo (↷):** Restore last undone operation
- **Stack Size:** Up to 50 operations stored

### Save & Export

- **Save Button (💾):** Export edited image to `./screenshots/`
- **Filename Format:** `edited_YYYYMMDD_HHMMSS.png`
- **Location:** All images saved in `./screenshots/` directory

### Return to Camera

- **Back Button (← Back to Camera):** Exit editing mode
- **State:** Returns to live camera feed
- **Capture Ready:** Can take new screenshots immediately

## Example Workflows

### Workflow 1: Brighten & Save

```
1. Capture screenshot with rectangle gesture
2. Move Brightness slider to +50
3. Click "💾 Save"
4. Image saved as edited_*.png
5. Click "← Back to Camera"
```

### Workflow 2: Grayscale Photo

```
1. Capture screenshot
2. Click "Grayscale" filter
3. Move Contrast slider to +100 for detail
4. Click "💾 Save"
5. Return to camera
```

### Workflow 3: Fix Blurry Image

```
1. Capture blurry screenshot
2. Move Sharpen slider to +30
3. Move Contrast slider to +80
4. Click "💾 Save"
5. Back to camera
```

### Workflow 4: Vintage Effect

```
1. Capture screenshot
2. Click "Sepia" filter
3. Move Contrast slider to -20
4. Move Brightness slider to +20
5. Click "💾 Save"
```

## Tips & Tricks

### Tip 1: Real-time Preview

- All sliders update the image in **real-time**
- See changes instantly as you adjust
- No "Apply" button needed

### Tip 2: Undo Liberally

- Use undo to experiment with settings
- Up to 50 changes can be undone
- Press undo multiple times to revert to original

### Tip 3: Combine Effects

- Brightness + Contrast + Sharpen = enhanced photo
- Grayscale + Sepia = artistic effect
- Blur + Sharpen = artistic blur effect

### Tip 4: Professional Results

- Start with contrast adjustment (+30 to +50)
- Add brightness (+10 to +30) if needed
- Sharpen by +15 to +30 for clarity
- Save and compare with original

### Tip 5: Slider Precision

- Hold and drag slowly for precise control
- Use keyboard arrows for fine adjustments
- Tick marks show 10-unit increments

## Keyboard Shortcuts

Currently not implemented, but planned for future:

- `Ctrl+S` - Save
- `Ctrl+Z` - Undo
- `Ctrl+Y` - Redo
- `Esc` - Back to Camera

## Troubleshooting

### Image Won't Load

- **Problem:** Screenshot captured but image not showing
- **Solution:** Check `./screenshots/` directory exists
- **Fix:** Create directory manually: `mkdir -p screenshots`

### No Preview Updates

- **Problem:** Slider moves but image doesn't change
- **Solution:** Check image was captured correctly
- **Fix:** Verify file in `./screenshots/` directory

### Save Button Disabled

- **Problem:** Can't click save button
- **Solution:** Could be displaying empty image
- **Fix:** Go back and capture new screenshot

### Editing Mode Won't Exit

- **Problem:** Back button not working
- **Solution:** Check for modal dialog blocking UI
- **Fix:** Close any dialogs, try back button again

## Supported Image Formats

**Input (Capture):**

- From camera as screenshot (auto PNG)

**Output (Save):**

- PNG (lossless, recommended)
- Quality: Full 8-bit color depth

**Image Size:**

- Depends on rectangle size
- Typically 200×200 to 1000×1000 pixels
- No size limits, but larger images = slower edits

## Performance Notes

- **Slider Responsiveness:** <100ms update
- **Filter Application:** 5-15ms typical
- **Save Operation:** 50-200ms depending on size
- **Undo/Redo:** <10ms instant

## Future Features (Roadmap)

Planned but not yet implemented:

- Gesture-based editing (hand position controls)
- Rotation slider
- Scale/zoom slider
- Custom color adjustments
- Red/Green/Blue channel adjustment
- Levels and curves adjustment
- Noise reduction
- White balance correction
- Side-by-side before/after view
- Batch processing multiple images
- Export with different quality levels

## Getting Help

If you encounter issues:

1. Check application logs for errors
2. Verify camera access permissions
3. Ensure `./screenshots/` directory exists
4. Try restarting the application
5. Test with different image sizes

## Best Practices

✅ **Do:**

- Start with small adjustments
- Use undo to experiment
- Save frequently
- Test filters on test images first

❌ **Don't:**

- Apply extreme slider values immediately
- Skip saving if you like a result
- Forget landmark visibility in rectangle formation
- Apply too many filters without checking

## Advanced Tips

### Combining Adjustments

```
Professional Enhancement Sequence:
1. Brightness: -10 to +20 (based on lighting)
2. Contrast: +30 to +50 (adds punch)
3. Sharpen: +15 to +25 (adds detail)
4. Export
```

### Artistic Effects

```
Vintage Photo Effect:
1. Apply Sepia filter
2. Brightness: +10
3. Contrast: -20
4. Blur: +5
```

### Restoration

```
Blurry Photo Fix:
1. Sharpen: +30 to +40
2. Contrast: +40 to +50
3. Brightness: +10
```

## Version History

**Current Version:** 1.0 (Release)

- ✅ Screenshot capture with rectangle gesture
- ✅ Brightness/Contrast adjustment
- ✅ Blur/Sharpen filters
- ✅ Grayscale/Sepia/Histogram equalization
- ✅ Undo/Redo (50-state stack)
- ✅ Save with timestamp
- ✅ Real-time preview

**Planned for v1.1:**

- Gesture-based editing controls
- Additional filters
- Preset effects
- Before/after preview

## Support

For issues or feature requests, check the project documentation or GitHub issues.
