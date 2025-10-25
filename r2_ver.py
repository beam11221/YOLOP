"""
Cloudflare R2 Upload/Download/Delete Script - OPTIMIZED VERSION
With concurrent uploads, downloads, and deletions for better performance
"""

import boto3
from botocore.exceptions import ClientError
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time


class CloudflareR2Optimized:
    """Optimized R2 client with concurrent upload/download/delete capabilities"""
    
    def __init__(self, account_id, access_key_id, secret_access_key, bucket_name, max_workers=8):
        """
        Initialize R2 client with concurrency support
        
        Args:
            account_id: Your Cloudflare account ID
            access_key_id: R2 access key ID
            secret_access_key: R2 secret access key
            bucket_name: Name of your R2 bucket
            max_workers: Maximum concurrent upload/download threads (default: 8)
        """
        self.bucket_name = bucket_name
        self.max_workers = max_workers
        self.stats_lock = Lock()  # Thread-safe stats updates
        
        # Create S3 client configured for Cloudflare R2
        self.s3_client = boto3.client(
            's3',
            endpoint_url=f'https://{account_id}.r2.cloudflarestorage.com',
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            region_name='auto'
        )
    
    def upload_file(self, local_file_path, r2_object_name=None):
        """Upload a single file to R2"""
        if r2_object_name is None:
            r2_object_name = os.path.basename(local_file_path)
        
        try:
            self.s3_client.upload_file(local_file_path, self.bucket_name, r2_object_name)
            return True, local_file_path, r2_object_name, os.path.getsize(local_file_path)
        except Exception as e:
            return False, local_file_path, r2_object_name, 0
    
    def _upload_single_file(self, local_path, r2_object_name):
        """Internal method for thread-safe single file upload"""
        try:
            file_size = os.path.getsize(local_path)
            self.s3_client.upload_file(local_path, self.bucket_name, r2_object_name)
            return {
                'success': True,
                'local_path': local_path,
                'r2_path': r2_object_name,
                'size': file_size,
                'error': None
            }
        except Exception as e:
            return {
                'success': False,
                'local_path': local_path,
                'r2_path': r2_object_name,
                'size': 0,
                'error': str(e)
            }
    
    def upload_directory_concurrent(self, local_directory, r2_prefix='', 
                                   exclude_patterns=None, show_progress=True):
        """
        Upload an entire directory to R2 with CONCURRENT uploads for better performance
        
        Args:
            local_directory: Path to the local directory
            r2_prefix: Optional prefix (folder path) in R2 bucket
            exclude_patterns: List of patterns to exclude
            show_progress: Show upload progress
            
        Returns:
            dict: Statistics about the upload
        """
        import fnmatch
        
        if not os.path.isdir(local_directory):
            print(f"✗ Error: {local_directory} is not a directory")
            return {'success_count': 0, 'fail_count': 0, 'total_size': 0}
        
        exclude_patterns = exclude_patterns or []
        
        # Collect all files to upload
        files_to_upload = []
        for root, dirs, files in os.walk(local_directory):
            for file in files:
                local_path = os.path.join(root, file)
                
                # Check exclude patterns
                should_exclude = False
                for pattern in exclude_patterns:
                    if fnmatch.fnmatch(file, pattern) or fnmatch.fnmatch(local_path, pattern):
                        should_exclude = True
                        break
                
                if should_exclude:
                    continue
                
                # Calculate relative path and R2 object name
                relative_path = os.path.relpath(local_path, local_directory)
                if r2_prefix:
                    prefix = r2_prefix.rstrip('/') + '/'
                    r2_object_name = prefix + relative_path.replace(os.sep, '/')
                else:
                    r2_object_name = relative_path.replace(os.sep, '/')
                
                files_to_upload.append((local_path, r2_object_name))
        
        if not files_to_upload:
            print("No files to upload")
            return {'success_count': 0, 'fail_count': 0, 'total_size': 0}
        
        print(f"\n📁 Uploading directory: {local_directory}")
        print(f"📍 R2 prefix: {r2_prefix or '(root)'}")
        print(f"🔢 Files to upload: {len(files_to_upload)}")
        print(f"⚡ Concurrent workers: {self.max_workers}")
        print(f"\nStarting concurrent upload...\n")
        
        # Upload files concurrently
        stats = {
            'success_count': 0,
            'fail_count': 0,
            'total_size': 0,
            'files': [],
            'failed_files': []
        }
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all upload tasks
            future_to_file = {
                executor.submit(self._upload_single_file, local_path, r2_path): (local_path, r2_path)
                for local_path, r2_path in files_to_upload
            }
            
            # Process completed uploads
            completed = 0
            for future in as_completed(future_to_file):
                result = future.result()
                completed += 1
                
                if result['success']:
                    stats['success_count'] += 1
                    stats['total_size'] += result['size']
                    stats['files'].append(result['r2_path'])
                    
                    if show_progress:
                        relative = os.path.relpath(result['local_path'], local_directory)
                        size_str = self._format_size(result['size'])
                        progress = f"[{completed}/{len(files_to_upload)}]"
                        print(f"  ✓ {progress} {relative} ({size_str})")
                else:
                    stats['fail_count'] += 1
                    stats['failed_files'].append({
                        'file': result['local_path'],
                        'error': result['error']
                    })
                    
                    if show_progress:
                        relative = os.path.relpath(result['local_path'], local_directory)
                        print(f"  ✗ {relative} - {result['error']}")
        
        elapsed_time = time.time() - start_time
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"📊 Upload Summary:")
        print(f"  ✓ Successful: {stats['success_count']} files")
        print(f"  ✗ Failed: {stats['fail_count']} files")
        print(f"  📦 Total size: {self._format_size(stats['total_size'])}")
        print(f"  ⏱️  Time: {elapsed_time:.2f} seconds")
        
        if stats['total_size'] > 0 and elapsed_time > 0:
            speed = stats['total_size'] / elapsed_time / (1024 * 1024)  # MB/s
            print(f"  ⚡ Speed: {speed:.2f} MB/s")
        
        print(f"{'='*60}\n")
        
        return stats
    
    def download_directory_concurrent(self, r2_prefix, local_directory, 
                                     create_dirs=True, show_progress=True):
        """
        Download all objects with a given prefix from R2 concurrently
        
        Args:
            r2_prefix: Prefix (folder path) in R2 bucket
            local_directory: Path to the local directory
            create_dirs: Whether to create local directories if they don't exist
            show_progress: Show download progress
            
        Returns:
            dict: Statistics about the download
        """
        stats = {
            'success_count': 0,
            'fail_count': 0,
            'total_size': 0,
            'files': [],
            'failed_files': []
        }
        
        print(f"\n📁 Downloading from R2 prefix: {r2_prefix}")
        print(f"📍 Local directory: {local_directory}")
        print(f"⚡ Concurrent workers: {self.max_workers}")
        
        # List all objects with the prefix
        try:
            objects_to_download = []
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=r2_prefix)
            
            for page in pages:
                if 'Contents' not in page:
                    continue
                
                for obj in page['Contents']:
                    r2_object_name = obj['Key']
                    file_size = obj['Size']
                    
                    # Calculate local path
                    relative_path = r2_object_name[len(r2_prefix):].lstrip('/')
                    local_path = os.path.join(local_directory, relative_path)
                    
                    objects_to_download.append((r2_object_name, local_path, file_size))
            
            if not objects_to_download:
                print("No objects found to download")
                return stats
            
            print(f"🔢 Files to download: {len(objects_to_download)}")
            print(f"\nStarting concurrent download...\n")
            
            start_time = time.time()
            
            def download_single(r2_path, local_path, size):
                try:
                    # Create directory if needed
                    if create_dirs:
                        os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    
                    self.s3_client.download_file(self.bucket_name, r2_path, local_path)
                    return {
                        'success': True,
                        'r2_path': r2_path,
                        'local_path': local_path,
                        'size': size,
                        'error': None
                    }
                except Exception as e:
                    return {
                        'success': False,
                        'r2_path': r2_path,
                        'local_path': local_path,
                        'size': 0,
                        'error': str(e)
                    }
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_file = {
                    executor.submit(download_single, r2_path, local_path, size): (r2_path, local_path)
                    for r2_path, local_path, size in objects_to_download
                }
                
                completed = 0
                for future in as_completed(future_to_file):
                    result = future.result()
                    completed += 1
                    
                    if result['success']:
                        stats['success_count'] += 1
                        stats['total_size'] += result['size']
                        stats['files'].append(result['local_path'])
                        
                        if show_progress:
                            relative = os.path.relpath(result['local_path'], local_directory)
                            size_str = self._format_size(result['size'])
                            progress = f"[{completed}/{len(objects_to_download)}]"
                            print(f"  ✓ {progress} {relative} ({size_str})")
                    else:
                        stats['fail_count'] += 1
                        stats['failed_files'].append({
                            'file': result['r2_path'],
                            'error': result['error']
                        })
                        
                        if show_progress:
                            print(f"  ✗ {result['r2_path']} - {result['error']}")
            
            elapsed_time = time.time() - start_time
            
            # Print summary
            print(f"\n{'='*60}")
            print(f"📊 Download Summary:")
            print(f"  ✓ Successful: {stats['success_count']} files")
            print(f"  ✗ Failed: {stats['fail_count']} files")
            print(f"  📦 Total size: {self._format_size(stats['total_size'])}")
            print(f"  ⏱️  Time: {elapsed_time:.2f} seconds")
            
            if stats['total_size'] > 0 and elapsed_time > 0:
                speed = stats['total_size'] / elapsed_time / (1024 * 1024)  # MB/s
                print(f"  ⚡ Speed: {speed:.2f} MB/s")
            
            print(f"{'='*60}\n")
            
        except ClientError as e:
            print(f"✗ Error listing objects: {e}")
        
        return stats
    
    def _delete_single_object(self, r2_object_name):
        """Internal method for thread-safe single object deletion"""
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=r2_object_name)
            return {
                'success': True,
                'r2_path': r2_object_name,
                'error': None
            }
        except Exception as e:
            return {
                'success': False,
                'r2_path': r2_object_name,
                'error': str(e)
            }
    
    def delete_directory_concurrent(self, r2_prefix, show_progress=True, dry_run=False):
        """
        Delete all objects with a given prefix (directory) from R2 concurrently
        
        Args:
            r2_prefix: Prefix (folder path) in R2 bucket to delete
            show_progress: Show deletion progress
            dry_run: If True, only list files that would be deleted without actually deleting
            
        Returns:
            dict: Statistics about the deletion
        """
        stats = {
            'success_count': 0,
            'fail_count': 0,
            'total_size': 0,
            'deleted_files': [],
            'failed_files': []
        }
        
        print(f"\n🗑️  {'DRY RUN - ' if dry_run else ''}Deleting R2 directory: {r2_prefix}")
        print(f"⚡ Concurrent workers: {self.max_workers}")
        
        # List all objects with the prefix
        try:
            objects_to_delete = []
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=r2_prefix)
            
            print(f"\n📋 Listing objects to delete...")
            
            for page in pages:
                if 'Contents' not in page:
                    continue
                
                for obj in page['Contents']:
                    r2_object_name = obj['Key']
                    file_size = obj['Size']
                    objects_to_delete.append((r2_object_name, file_size))
                    stats['total_size'] += file_size
            
            if not objects_to_delete:
                print("✓ No objects found with this prefix")
                return stats
            
            print(f"🔢 Objects to delete: {len(objects_to_delete)}")
            print(f"📦 Total size: {self._format_size(stats['total_size'])}")
            
            if dry_run:
                print(f"\n⚠️  DRY RUN MODE - Files that would be deleted:\n")
                for r2_path, size in objects_to_delete:
                    size_str = self._format_size(size)
                    print(f"  • {r2_path} ({size_str})")
                print(f"\n{'='*60}")
                print(f"📊 Dry Run Summary:")
                print(f"  🔢 Objects that would be deleted: {len(objects_to_delete)}")
                print(f"  📦 Total size: {self._format_size(stats['total_size'])}")
                print(f"{'='*60}\n")
                return stats
            
            # Confirm deletion
            print(f"\n⚠️  WARNING: This will permanently delete {len(objects_to_delete)} objects!")
            print(f"⚠️  Total size: {self._format_size(stats['total_size'])}")
            response = input(f"\nType 'DELETE' to confirm deletion: ")
            
            if response != 'DELETE':
                print("❌ Deletion cancelled")
                return stats
            
            print(f"\n🗑️  Starting concurrent deletion...\n")
            
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all deletion tasks
                future_to_file = {
                    executor.submit(self._delete_single_object, r2_path): (r2_path, size)
                    for r2_path, size in objects_to_delete
                }
                
                # Process completed deletions
                completed = 0
                for future in as_completed(future_to_file):
                    result = future.result()
                    r2_path, size = future_to_file[future]
                    completed += 1
                    
                    if result['success']:
                        stats['success_count'] += 1
                        stats['deleted_files'].append(result['r2_path'])
                        
                        if show_progress:
                            size_str = self._format_size(size)
                            progress = f"[{completed}/{len(objects_to_delete)}]"
                            print(f"  ✓ {progress} Deleted: {result['r2_path']} ({size_str})")
                    else:
                        stats['fail_count'] += 1
                        stats['failed_files'].append({
                            'file': result['r2_path'],
                            'error': result['error']
                        })
                        
                        if show_progress:
                            print(f"  ✗ Failed: {result['r2_path']} - {result['error']}")
            
            elapsed_time = time.time() - start_time
            
            # Print summary
            print(f"\n{'='*60}")
            print(f"📊 Deletion Summary:")
            print(f"  ✓ Successfully deleted: {stats['success_count']} objects")
            print(f"  ✗ Failed: {stats['fail_count']} objects")
            print(f"  📦 Total size deleted: {self._format_size(stats['total_size'])}")
            print(f"  ⏱️  Time: {elapsed_time:.2f} seconds")
            
            if stats['success_count'] > 0:
                rate = stats['success_count'] / elapsed_time
                print(f"  ⚡ Deletion rate: {rate:.2f} objects/second")
            
            print(f"{'='*60}\n")
            
        except ClientError as e:
            print(f"✗ Error listing/deleting objects: {e}")
        
        return stats
    
    @staticmethod
    def _format_size(size_bytes):
        """Format bytes to human readable size"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} PB"


def main():
    """Example usage of optimized R2 client"""
    import os
    
    # Configuration
    # ACCOUNT_ID = os.getenv('R2_ACCOUNT_ID', 'your_account_id')
    # ACCESS_KEY_ID = os.getenv('R2_ACCESS_KEY_ID', 'your_access_key_id')
    # SECRET_ACCESS_KEY = os.getenv('R2_SECRET_ACCESS_KEY', 'your_secret_access_key')
    # BUCKET_NAME = os.getenv('R2_BUCKET_NAME', 'your_bucket_name')
    
    ACCOUNT_ID = "e0c48e6a3d8331671bbc8f715a7c3483"
    ACCESS_KEY_ID = "f648ee228def753fa2d7538782e684cd"
    SECRET_ACCESS_KEY = "0011640d7a66b1a27b3461a6bf29a8387e66c17377b1d91c514e3bdda6202fa9"
    BUCKET_NAME = "master"

    # Initialize with more workers for better performance
    r2 = CloudflareR2Optimized(
        account_id=ACCOUNT_ID,
        access_key_id=ACCESS_KEY_ID,
        secret_access_key=SECRET_ACCESS_KEY,
        bucket_name=BUCKET_NAME,
        max_workers=16  # Increase for more concurrency (default: 8)
    )
    
    # Example 1: Upload directory with concurrent uploads
    print("="*60)
    print("CONCURRENT DIRECTORY UPLOAD")
    print("="*60)
    
    stats = r2.upload_directory_concurrent(
        local_directory='/workspace/YOLOP/runs/BddDataset',
        r2_prefix='ver1/BddDataset_yolop',
        exclude_patterns=['*.pyc', '__pycache__', '.git', '*.tmp'],
        show_progress=True
    )
    
    print(f"\n✅ Upload completed!")
    print(f"   Files uploaded: {stats['success_count']}")
    print(f"   Total size: {CloudflareR2Optimized._format_size(stats['total_size'])}")
    
    # # Example 2: Download directory with concurrent downloads
    # print("\n" + "="*60)
    # print("CONCURRENT DIRECTORY DOWNLOAD")
    # print("="*60)
    
    # stats = r2.download_directory_concurrent(
    #     r2_prefix='uploads/my_project',
    #     local_directory='downloaded_project',
    #     show_progress=True
    # )
    
    # print(f"\n✅ Download completed!")
    # print(f"   Files downloaded: {stats['success_count']}")
    # print(f"   Total size: {CloudflareR2Optimized._format_size(stats['total_size'])}")
    
    # # Example 3: Delete directory (with dry run first)
    # print("\n" + "="*60)
    # print("DELETE DIRECTORY (DRY RUN)")
    # print("="*60)
    
    # # First do a dry run to see what would be deleted
    # stats = r2.delete_directory_concurrent(
    #     r2_prefix='uploads/old_project',
    #     show_progress=True,
    #     dry_run=True  # This will only list files without deleting
    # )
    
    # Example 4: Actually delete directory
    # print("\n" + "="*60)
    # print("DELETE DIRECTORY (ACTUAL)")
    # print("="*60)
    
    # stats = r2.delete_directory_concurrent(
    #     r2_prefix='uploads/',
    #     show_progress=True,
    #     dry_run=False  # This will actually delete files (requires confirmation)
    # )
    
    # print(f"\n✅ Deletion completed!")
    # print(f"   Files deleted: {stats['success_count']}")
    # print(f"   Total size: {CloudflareR2Optimized._format_size(stats['total_size'])}")


if __name__ == "__main__":
    main()